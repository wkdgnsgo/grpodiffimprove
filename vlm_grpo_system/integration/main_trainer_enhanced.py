"""
Enhanced VLM GRPO Main Trainer
==============================

MS Swift CoZ GRPO를 참조하여 개선된 통합 VLM GRPO 시스템입니다.
LoRA 버전과 전체 학습 버전을 모두 지원합니다.

주요 개선사항:
1. MS Swift 스타일 설정 지원
2. LoRA/QLoRA/Full 학습 모드 선택
3. 자동 디바이스 최적화 (MPS/CUDA/CPU)
4. 메모리 효율적 학습
5. 실시간 성능 모니터링
6. Wandb 통합

사용법:
    # LoRA 버전 (메모리 효율적)
    trainer = EnhancedVLMGRPOSystem(train_type="lora", lora_rank=8)
    
    # 전체 학습 버전 (고성능)
    trainer = EnhancedVLMGRPOSystem(train_type="full", use_deepspeed=True)

Author: AI Assistant (Based on MS Swift CoZ GRPO)
Date: 2025-01-22
"""

import torch
import torch.nn as nn
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field
import numpy as np

# 기존 모듈들 임포트
from ..models.vlm_wrapper import VLMWrapper
from ..models.sd_generator import SD3Generator
from ..models.clip_reward import CLIPRewardCalculator
from ..training.grpo_trainer import GRPOTrainer, GRPOConfig
from ..utils.data_loader import VLMDataLoader
from ..evaluation.validator import VLMValidator
from ..integration.wandb_logger import WandbLogger

# LoRA 관련 임포트 (선택적)
try:
    from peft import (
        LoraConfig, 
        get_peft_model, 
        TaskType,
        PeftModel,
        prepare_model_for_kbit_training
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class EnhancedVLMGRPOConfig:
    """
    개선된 VLM GRPO 시스템 설정
    
    MS Swift 스타일 설정을 포함하여 LoRA와 전체 학습을 모두 지원합니다.
    """
    # 기본 모델 설정
    vlm_model: str = "microsoft/DialoGPT-medium"
    sd_model: str = "stabilityai/stable-diffusion-3-medium"
    clip_model: str = "openai/clip-vit-base-patch32"
    
    # 학습 타입 설정 (MS Swift 스타일)
    train_type: Literal["full", "lora", "qlora"] = "lora"
    
    # LoRA 설정 (MS Swift 기본값)
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: str = "all-linear"  # MS Swift 스타일
    
    # 학습 하이퍼파라미터
    learning_rate: float = 1e-5
    num_iterations: int = 20
    group_size: int = 4
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.05
    
    # GRPO 특화 설정
    num_generations: int = 4
    temperature: float = 0.8
    top_p: float = 0.9
    grpo_epochs: int = 2
    kl_beta: float = 0.01
    clip_epsilon: float = 0.2
    
    # 데이터 설정
    train_data_path: str = "train_prompts.jsonl"
    val_data_path: str = "val_prompts.jsonl"
    max_length: int = 2048
    max_completion_length: int = 1024
    
    # 평가 및 저장 설정
    validation_interval: int = 5
    checkpoint_interval: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2
    logging_steps: int = 5
    
    # 출력 설정
    output_dir: str = "vlm_grpo_results"
    log_completions: bool = True
    
    # 분산 학습 설정
    use_deepspeed: bool = False
    deepspeed_config: str = "zero2"  # "zero2", "zero3"
    
    # 하드웨어 최적화
    device: str = "auto"
    torch_dtype: str = "bfloat16"
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    
    # 보상 함수 설정
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "clip_similarity": 0.6,
        "image_quality": 0.3,
        "semantic_consistency": 0.1
    })
    
    # 실험 추적
    use_wandb: bool = True
    wandb_project: str = "vlm-grpo-enhanced"
    wandb_run_name: Optional[str] = None
    
    # 시스템 프롬프트
    system_prompt: Optional[str] = None

class EnhancedVLMGRPOSystem:
    """
    개선된 VLM GRPO 통합 시스템
    
    MS Swift CoZ GRPO를 참조하여 구현된 개선된 시스템으로,
    LoRA와 전체 학습을 모두 지원합니다.
    
    주요 특징:
    1. MS Swift 스타일 설정
    2. 자동 메모리 최적화
    3. 유연한 학습 모드 선택
    4. 실시간 성능 모니터링
    5. 완전한 실험 추적
    """
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Enhanced VLM GRPO System 초기화
        
        Args:
            config_path (Optional[str]): 설정 파일 경로
            **kwargs: 직접 설정 오버라이드
        """
        # 설정 로드
        self.config = self._load_config(config_path, **kwargs)
        
        # 컴포넌트 초기화
        self.vlm = None
        self.sd_generator = None
        self.clip_calculator = None
        self.grpo_trainer = None
        self.data_loader = None
        self.validator = None
        self.wandb_logger = None
        
        # 학습 상태
        self.training_stats = {
            'current_iteration': 0,
            'total_samples': 0,
            'best_reward': 0.0,
            'training_time': 0.0
        }
        
        # 디바이스 설정
        self._setup_device()
        
        logger.info("🚀 Enhanced VLM GRPO System Initialized")
        self._print_system_info()
    
    def _load_config(self, config_path: Optional[str], **kwargs) -> EnhancedVLMGRPOConfig:
        """설정 로드 및 오버라이드 적용"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config = EnhancedVLMGRPOConfig(**config_dict)
        else:
            config = EnhancedVLMGRPOConfig()
        
        # kwargs로 오버라이드
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def _setup_device(self):
        """디바이스 자동 설정"""
        if self.config.device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("🍎 Using Apple Silicon MPS")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("🚀 Using CUDA GPU")
            else:
                self.device = torch.device("cpu")
                logger.info("💻 Using CPU")
        else:
            self.device = torch.device(self.config.device)
    
    def _print_system_info(self):
        """시스템 정보 출력"""
        logger.info("=" * 70)
        logger.info("🔧 Enhanced VLM GRPO System Configuration")
        logger.info("=" * 70)
        logger.info(f"📋 Training Configuration:")
        logger.info(f"   - Train Type: {self.config.train_type}")
        logger.info(f"   - VLM Model: {self.config.vlm_model}")
        logger.info(f"   - Device: {self.device}")
        logger.info(f"   - Torch Dtype: {self.config.torch_dtype}")
        
        if self.config.train_type in ["lora", "qlora"]:
            logger.info(f"🎯 LoRA Configuration:")
            logger.info(f"   - LoRA Rank: {self.config.lora_rank}")
            logger.info(f"   - LoRA Alpha: {self.config.lora_alpha}")
            logger.info(f"   - Target Modules: {self.config.target_modules}")
        
        logger.info(f"⚙️ Training Parameters:")
        logger.info(f"   - Learning Rate: {self.config.learning_rate}")
        logger.info(f"   - Batch Size: {self.config.per_device_train_batch_size}")
        logger.info(f"   - Num Iterations: {self.config.num_iterations}")
        logger.info(f"   - Group Size: {self.config.group_size}")
        
        if self.config.use_deepspeed:
            logger.info(f"⚡ DeepSpeed: {self.config.deepspeed_config}")
        
        logger.info(f"📊 Experiment Tracking:")
        logger.info(f"   - Wandb: {self.config.use_wandb}")
        logger.info(f"   - Output Dir: {self.config.output_dir}")
        logger.info("=" * 70)
    
    def initialize_components(self):
        """모든 컴포넌트 초기화"""
        logger.info("🔧 Initializing system components...")
        
        try:
            # 1. VLM 초기화 (LoRA 또는 전체 학습)
            self._initialize_vlm()
            
            # 2. SD3 Generator 초기화
            self._initialize_sd_generator()
            
            # 3. CLIP Reward Calculator 초기화
            self._initialize_clip_calculator()
            
            # 4. GRPO Trainer 초기화
            self._initialize_grpo_trainer()
            
            # 5. Data Loader 초기화
            self._initialize_data_loader()
            
            # 6. Validator 초기화
            self._initialize_validator()
            
            # 7. Wandb Logger 초기화
            if self.config.use_wandb:
                self._initialize_wandb_logger()
            
            logger.info("✅ All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    def _initialize_vlm(self):
        """VLM 초기화 (LoRA 또는 전체 학습 지원)"""
        logger.info(f"📥 Initializing VLM: {self.config.vlm_model}")
        
        # 기본 VLM 래퍼 생성
        self.vlm = VLMWrapper(
            model_name=self.config.vlm_model,
            device=str(self.device),
            max_new_tokens=self.config.max_completion_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p
        )
        
        # LoRA 설정 적용
        if self.config.train_type in ["lora", "qlora"]:
            self._apply_lora_to_vlm()
        
        logger.info(f"✅ VLM initialized with {self.config.train_type} training")
    
    def _apply_lora_to_vlm(self):
        """VLM에 LoRA 적용"""
        if not PEFT_AVAILABLE:
            logger.warning("⚠️ PEFT not available. Falling back to full training.")
            self.config.train_type = "full"
            return
        
        try:
            # LoRA 설정
            target_modules = self._get_target_modules()
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=target_modules,
                bias="none",
                inference_mode=False
            )
            
            # QLoRA 전처리
            if self.config.train_type == "qlora":
                self.vlm.model = prepare_model_for_kbit_training(
                    self.vlm.model,
                    use_gradient_checkpointing=self.config.gradient_checkpointing
                )
            
            # LoRA 모델 생성
            self.vlm.model = get_peft_model(self.vlm.model, lora_config)
            
            # 파라미터 통계
            trainable_params = sum(p.numel() for p in self.vlm.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.vlm.model.parameters())
            
            logger.info(f"🎯 LoRA applied successfully:")
            logger.info(f"   - Trainable params: {trainable_params:,}")
            logger.info(f"   - Total params: {total_params:,}")
            logger.info(f"   - Trainable ratio: {100 * trainable_params / total_params:.2f}%")
            
        except Exception as e:
            logger.error(f"❌ LoRA application failed: {e}")
            logger.warning("⚠️ Falling back to full training")
            self.config.train_type = "full"
    
    def _get_target_modules(self) -> List[str]:
        """MS Swift 스타일 target_modules 처리"""
        if self.config.target_modules == "all-linear":
            # 모든 선형 레이어 찾기
            linear_modules = []
            for name, module in self.vlm.model.named_modules():
                if isinstance(module, nn.Linear):
                    module_name = name.split('.')[-1]
                    if module_name not in linear_modules:
                        linear_modules.append(module_name)
            
            # 일반적인 Transformer 모듈들
            common_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "fc1", "fc2", "c_attn", "c_proj"
            ]
            
            # 실제 존재하는 모듈만 반환
            found_modules = [m for m in common_modules if m in linear_modules]
            return found_modules or linear_modules
        else:
            return [self.config.target_modules]
    
    def _initialize_sd_generator(self):
        """SD3 Generator 초기화"""
        logger.info(f"🎨 Initializing SD3 Generator: {self.config.sd_model}")
        
        self.sd_generator = SD3Generator(
            model_name=self.config.sd_model,
            device=str(self.device),
            torch_dtype=self.config.torch_dtype,
            enable_memory_efficient_attention=True,
            enable_attention_slicing=True
        )
        
        logger.info("✅ SD3 Generator initialized")
    
    def _initialize_clip_calculator(self):
        """CLIP Reward Calculator 초기화"""
        logger.info(f"🔍 Initializing CLIP Calculator: {self.config.clip_model}")
        
        self.clip_calculator = CLIPRewardCalculator(
            model_name=self.config.clip_model,
            device=str(self.device),
            reward_weights=self.config.reward_weights
        )
        
        logger.info("✅ CLIP Calculator initialized")
    
    def _initialize_grpo_trainer(self):
        """GRPO Trainer 초기화"""
        logger.info("🎯 Initializing GRPO Trainer...")
        
        # GRPO 설정 생성
        grpo_config = GRPOConfig(
            learning_rate=self.config.learning_rate,
            group_size=self.config.group_size,
            num_iterations=self.config.num_iterations,
            grpo_epochs=self.config.grpo_epochs,
            kl_beta=self.config.kl_beta,
            clip_epsilon=self.config.clip_epsilon,
            max_grad_norm=self.config.max_grad_norm,
            max_new_tokens=self.config.max_completion_length,
            temperature=self.config.temperature,
            device=str(self.device)
        )
        
        self.grpo_trainer = GRPOTrainer(
            vlm_model=self.vlm,
            config=grpo_config
        )
        
        logger.info("✅ GRPO Trainer initialized")
    
    def _initialize_data_loader(self):
        """Data Loader 초기화"""
        logger.info("📊 Initializing Data Loader...")
        
        self.data_loader = VLMDataLoader(
            train_path=self.config.train_data_path,
            val_path=self.config.val_data_path,
            batch_size=self.config.per_device_train_batch_size
        )
        
        logger.info("✅ Data Loader initialized")
    
    def _initialize_validator(self):
        """Validator 초기화"""
        logger.info("🔬 Initializing Validator...")
        
        self.validator = VLMValidator(
            vlm=self.vlm,
            sd_generator=self.sd_generator,
            clip_calculator=self.clip_calculator,
            validation_interval=self.config.validation_interval
        )
        
        logger.info("✅ Validator initialized")
    
    def _initialize_wandb_logger(self):
        """Wandb Logger 초기화"""
        logger.info("📈 Initializing Wandb Logger...")
        
        # 실행 이름 생성
        run_name = self.config.wandb_run_name or f"vlm-grpo-{self.config.train_type}-{int(time.time())}"
        
        # Wandb 설정
        wandb_config = {
            "train_type": self.config.train_type,
            "vlm_model": self.config.vlm_model,
            "learning_rate": self.config.learning_rate,
            "group_size": self.config.group_size,
            "num_iterations": self.config.num_iterations,
            "device": str(self.device)
        }
        
        # LoRA 설정 추가
        if self.config.train_type in ["lora", "qlora"]:
            wandb_config.update({
                "lora_rank": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
                "target_modules": self.config.target_modules
            })
        
        self.wandb_logger = WandbLogger(
            project_name=self.config.wandb_project,
            run_name=run_name,
            config=wandb_config
        )
        
        logger.info("✅ Wandb Logger initialized")
    
    def run_training(self):
        """메인 학습 실행"""
        logger.info("🚀 Starting Enhanced VLM GRPO Training...")
        logger.info(f"   - Train Type: {self.config.train_type}")
        logger.info(f"   - Iterations: {self.config.num_iterations}")
        logger.info(f"   - Group Size: {self.config.group_size}")
        
        start_time = time.time()
        
        try:
            for iteration in range(1, self.config.num_iterations + 1):
                self.training_stats['current_iteration'] = iteration
                
                # 학습 배치 생성
                train_batch = self.data_loader.get_train_batch(
                    group_size=self.config.group_size
                )
                
                # GRPO 학습 스텝
                step_stats = self._run_training_step(train_batch)
                
                # 통계 업데이트
                self._update_training_stats(step_stats)
                
                # 로깅
                if iteration % self.config.logging_steps == 0:
                    self._log_training_progress(iteration, step_stats)
                
                # 검증
                if iteration % self.config.validation_interval == 0:
                    val_stats = self._run_validation()
                    self._log_validation_results(iteration, val_stats)
                
                # 체크포인트 저장
                if iteration % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(iteration)
                
                # Wandb 로깅
                if self.wandb_logger:
                    self._log_to_wandb(iteration, step_stats)
            
            # 최종 결과 저장
            self._save_final_results()
            
            total_time = time.time() - start_time
            self.training_stats['training_time'] = total_time
            
            logger.info("🎉 Training completed successfully!")
            logger.info(f"   - Total time: {total_time:.2f}s")
            logger.info(f"   - Best reward: {self.training_stats['best_reward']:.4f}")
            
        except KeyboardInterrupt:
            logger.info("⚠️ Training interrupted by user")
            self._save_checkpoint(self.training_stats['current_iteration'], final=True)
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
        
        finally:
            if self.wandb_logger:
                self.wandb_logger.finish()
    
    def _run_training_step(self, train_batch: List[str]) -> Dict[str, float]:
        """단일 학습 스텝 실행"""
        # GRPO 학습 스텝 실행
        step_stats = self.grpo_trainer.train_step(train_batch)
        
        # 샘플 수 업데이트
        self.training_stats['total_samples'] += len(train_batch)
        
        return step_stats
    
    def _update_training_stats(self, step_stats: Dict[str, float]):
        """학습 통계 업데이트"""
        if step_stats['avg_reward'] > self.training_stats['best_reward']:
            self.training_stats['best_reward'] = step_stats['avg_reward']
    
    def _log_training_progress(self, iteration: int, stats: Dict[str, float]):
        """학습 진행 상황 로깅"""
        logger.info(f"📊 Iteration {iteration}/{self.config.num_iterations}")
        logger.info(f"   - Avg Reward: {stats['avg_reward']:.4f}")
        logger.info(f"   - Policy Loss: {stats['policy_loss']:.4f}")
        logger.info(f"   - KL Divergence: {stats['kl_divergence']:.4f}")
        logger.info(f"   - Entropy: {stats['entropy']:.4f}")
        logger.info(f"   - Grad Norm: {stats['grad_norm']:.4f}")
    
    def _run_validation(self) -> Dict[str, Any]:
        """검증 실행"""
        logger.info("🔬 Running validation...")
        
        val_batch = self.data_loader.get_val_batch()
        val_stats = self.validator.validate(val_batch)
        
        logger.info(f"✅ Validation completed: {val_stats['success_rate']:.2%}")
        return val_stats
    
    def _log_validation_results(self, iteration: int, val_stats: Dict[str, Any]):
        """검증 결과 로깅"""
        logger.info(f"🔬 Validation Results (Iteration {iteration}):")
        logger.info(f"   - Success Rate: {val_stats['success_rate']:.2%}")
        logger.info(f"   - Avg Quality: {val_stats['avg_quality_score']:.4f}")
        logger.info(f"   - Avg Similarity: {val_stats['avg_similarity_score']:.4f}")
    
    def _save_checkpoint(self, iteration: int, final: bool = False):
        """체크포인트 저장"""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint_iter_{iteration}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 저장 (LoRA 또는 전체 모델)
        if self.config.train_type in ["lora", "qlora"]:
            # LoRA 어댑터만 저장
            self.vlm.model.save_pretrained(checkpoint_dir)
            logger.info(f"💾 LoRA checkpoint saved: {checkpoint_dir}")
        else:
            # 전체 모델 저장
            torch.save(self.vlm.model.state_dict(), checkpoint_dir / "model.pt")
            logger.info(f"💾 Full model checkpoint saved: {checkpoint_dir}")
        
        # 학습 통계 저장
        with open(checkpoint_dir / "training_stats.json", 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        # 설정 저장
        with open(checkpoint_dir / "config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        if final:
            logger.info(f"💾 Final checkpoint saved: {checkpoint_dir}")
    
    def _log_to_wandb(self, iteration: int, stats: Dict[str, float]):
        """Wandb 로깅"""
        if self.wandb_logger:
            log_data = {
                "iteration": iteration,
                "train/avg_reward": stats['avg_reward'],
                "train/policy_loss": stats['policy_loss'],
                "train/kl_divergence": stats['kl_divergence'],
                "train/entropy": stats['entropy'],
                "train/grad_norm": stats['grad_norm'],
                "system/total_samples": self.training_stats['total_samples']
            }
            
            self.wandb_logger.log_metrics(log_data)
    
    def _save_final_results(self):
        """최종 결과 저장"""
        results_dir = Path(self.config.output_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 최고 성능 모델 저장
        if self.config.train_type in ["lora", "qlora"]:
            self.vlm.model.save_pretrained(results_dir / "best_model")
        else:
            torch.save(self.vlm.model.state_dict(), results_dir / "best_model.pt")
        
        # 최종 통계 저장
        final_stats = {
            **self.training_stats,
            "config": self.config.__dict__,
            "train_type": self.config.train_type,
            "total_parameters": sum(p.numel() for p in self.vlm.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.vlm.model.parameters() if p.requires_grad)
        }
        
        with open(results_dir / "final_results.json", 'w') as f:
            json.dump(final_stats, f, indent=2, default=str)
        
        logger.info(f"💾 Final results saved to: {results_dir}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """학습 통계 반환"""
        return self.training_stats.copy()
    
    def print_system_summary(self):
        """시스템 요약 출력"""
        print("\n" + "=" * 70)
        print("🔍 Enhanced VLM GRPO System Summary")
        print("=" * 70)
        print(f"Train Type: {self.config.train_type}")
        print(f"VLM Model: {self.config.vlm_model}")
        print(f"Device: {self.device}")
        
        if self.vlm and hasattr(self.vlm, 'model'):
            total_params = sum(p.numel() for p in self.vlm.model.parameters())
            trainable_params = sum(p.numel() for p in self.vlm.model.parameters() if p.requires_grad)
            print(f"Total Parameters: {total_params:,}")
            print(f"Trainable Parameters: {trainable_params:,}")
            print(f"Trainable Ratio: {100 * trainable_params / total_params:.2f}%")
        
        if self.config.train_type in ["lora", "qlora"]:
            print(f"LoRA Rank: {self.config.lora_rank}")
            print(f"LoRA Alpha: {self.config.lora_alpha}")
        
        print(f"Learning Rate: {self.config.learning_rate}")
        print(f"Batch Size: {self.config.per_device_train_batch_size}")
        print(f"Iterations: {self.config.num_iterations}")
        print(f"Output Directory: {self.config.output_dir}")
        print("=" * 70)


# 편의 함수들
def create_lora_trainer(lora_rank: int = 8, lora_alpha: int = 32, **kwargs) -> EnhancedVLMGRPOSystem:
    """LoRA 트레이너 생성 편의 함수"""
    config = EnhancedVLMGRPOConfig(
        train_type="lora",
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        **kwargs
    )
    return EnhancedVLMGRPOSystem(config=config)

def create_full_trainer(use_deepspeed: bool = True, **kwargs) -> EnhancedVLMGRPOSystem:
    """전체 학습 트레이너 생성 편의 함수"""
    config = EnhancedVLMGRPOConfig(
        train_type="full",
        use_deepspeed=use_deepspeed,
        learning_rate=1e-6,  # 전체 학습은 더 작은 학습률 사용
        **kwargs
    )
    return EnhancedVLMGRPOSystem(config=config)


# 테스트 코드
if __name__ == "__main__":
    print("✅ Enhanced VLM GRPO System Ready!")
    print("\n📋 Available Training Modes:")
    print("1. LoRA Training (Memory Efficient)")
    print("   trainer = create_lora_trainer(lora_rank=8)")
    print("\n2. Full Training (High Performance)")
    print("   trainer = create_full_trainer(use_deepspeed=True)")
    print("\n3. Custom Configuration")
    print("   trainer = EnhancedVLMGRPOSystem(train_type='lora', lora_rank=16)") 