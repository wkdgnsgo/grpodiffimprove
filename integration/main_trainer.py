"""
VLM GRPO Main Trainer
====================

모든 컴포넌트를 통합하여 End-to-End VLM GRPO 학습을 수행하는 메인 모듈입니다.

시스템 구조:
User Prompt → VLM → Enhanced Prompt → SD3 → Image → CLIP Reward → GRPO Update

주요 기능:
1. 전체 시스템 초기화 및 설정
2. 학습 루프 실행
3. 검증 및 평가
4. 결과 저장 및 시각화
5. Wandb 통합

Author: AI Assistant
Date: 2025-01-22
"""

import os
import sys
import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import numpy as np

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

# 각 모듈 임포트
VLMWrapper = None
SD3Generator = None
CLIPRewardCalculator = None
MultiRewardCalculator = None
GRPOTrainer = None
GRPOConfig = None
PromptDataLoader = None
ValidationEvaluator = None
WandbLogger = None

try:
    from models.vlm_wrapper import VLMWrapper
except ImportError as e:
    print(f"⚠️ VLMWrapper import warning: {e}")

try:
    from models.sd_generator import SD3Generator  
except ImportError as e:
    print(f"⚠️ SD3Generator import warning: {e}")

try:
    from models.clip_reward import CLIPRewardCalculator, MultiRewardCalculator
except ImportError as e:
    print(f"⚠️ CLIP modules import warning: {e}")

try:
    from training.grpo_trainer import GRPOTrainer, GRPOConfig
except ImportError as e:
    print(f"⚠️ GRPO modules import warning: {e}")

try:
    from utils.data_loader import DataLoader as PromptDataLoader
except ImportError as e:
    print(f"⚠️ DataLoader import warning: {e}")

try:
    from evaluation.validator import ValidationEvaluator
except ImportError as e:
    print(f"⚠️ Validator import warning: {e}")

try:
    from integration.wandb_logger import WandbLogger
except ImportError as e:
    print(f"⚠️ WandbLogger import warning: {e}")

logger = logging.getLogger(__name__)

class VLMGRPOSystem:
    """
    VLM GRPO 전체 시스템을 관리하는 메인 클래스
    
    이 클래스는:
    1. 모든 컴포넌트 초기화 및 연결
    2. 전체 학습 파이프라인 실행
    3. 실시간 모니터링 및 로깅
    4. 체크포인트 관리
    5. 결과 분석 및 저장
    """
    
    def __init__(self, config_path: str = "config/default_config.json"):
        """
        VLM GRPO System 초기화
        
        Args:
            config_path (str): 설정 파일 경로
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # 로깅 설정
        self._setup_logging()
        
        # --- Core Components ---
        self.vlm_policy = None              # VLM 정책 네트워크 (Qwen2.5-VL)
        self.sd_generator = None            # SD3 생성기 (동결됨)
        self.clip_reward = None             # CLIP 보상 계산기 (동결됨)
        self.grpo_trainer = None            # GRPO 트레이너
        self.data_loader = None             # 데이터 로더
        self.validator = None               # 검증기
        self.wandb_logger = None            # Wandb 로거
        
        # --- Training Configuration ---
        self.grpo_config = None
        
        # --- Lists for Logging/Plotting ---
        self.iteration_rewards = []
        self.iteration_policy_losses = []
        self.iteration_entropies = []
        self.iteration_kl_divs = []
        
        # --- Training Statistics ---
        self.training_stats = {
            'best_reward': float('-inf'),
            'total_iterations': 0,
            'total_time': 0.0,
            'current_iteration': 0
        }
        
        logger.info("🚀 VLM GRPO System initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"📄 Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            raise
    
    def _setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('vlm_grpo_training.log')
            ]
        )
    
    def initialize_components(self):
        """
        모든 컴포넌트 초기화 - 제공된 코드 형식을 따름
        
        # --- Initialization ---
        vlm_policy: VLMWrapper = VLMWrapper(...)
        sd_generator: SD3Generator = SD3Generator(...)
        clip_reward: CLIPRewardCalculator = CLIPRewardCalculator(...)
        grpo_trainer: GRPOTrainer = GRPOTrainer(vlm_policy, sd_generator, clip_reward, config)
        """
        try:
            logger.info("🔧 Initializing components...")
            
            # --- VLM Policy Network Initialization ---
            logger.info("📝 Initializing VLM Policy Network...")
            if VLMWrapper is None:
                raise ImportError("VLMWrapper not available. Please install required dependencies.")
            
            self.vlm_policy: VLMWrapper = VLMWrapper(
                config_path=self.config_path,
                device=self.config['system_settings']['device'],
                max_new_tokens=self.config['generation_settings']['vlm_generation']['max_new_tokens'],
                temperature=self.config['generation_settings']['vlm_generation']['temperature'],
                top_p=self.config['generation_settings']['vlm_generation']['top_p']
            )
            logger.info(f"✅ VLM Policy Network initialized: {self.vlm_policy.model_name}")
            
            # --- SD3 Generator Initialization (Frozen) ---
            logger.info("🎨 Initializing SD3 Generator...")
            if SD3Generator is None:
                raise ImportError("SD3Generator not available. Please install required dependencies.")
            
            self.sd_generator: SD3Generator = SD3Generator(
                config_path=self.config_path,
                device=self.config['system_settings']['device'],
                height=self.config['generation_settings']['sd_generation']['height'],
                width=self.config['generation_settings']['sd_generation']['width'],
                num_inference_steps=self.config['generation_settings']['sd_generation']['num_inference_steps'],
                guidance_scale=self.config['generation_settings']['sd_generation']['guidance_scale']
            )
            logger.info(f"✅ SD3 Generator initialized: {self.sd_generator.model_name}")
            
            # --- CLIP Reward Calculator Initialization (Frozen) ---
            logger.info("🏆 Initializing CLIP Reward Calculator...")
            if CLIPRewardCalculator is None:
                raise ImportError("CLIPRewardCalculator not available. Please install required dependencies.")
            
            self.clip_reward: CLIPRewardCalculator = CLIPRewardCalculator(
                model_name=self.config['model_settings']['clip_model'],
                device=self.config['system_settings']['device'],
                reward_weights=self.config['reward_settings']['reward_weights']
            )
            logger.info(f"✅ CLIP Reward Calculator initialized: {self.clip_reward.model_name}")
            
            # --- GRPO Configuration ---
            self.grpo_config: GRPOConfig = GRPOConfig(
                learning_rate=self.config['training_settings']['learning_rate'],
                group_size=self.config['training_settings']['group_size'],
                num_iterations=self.config['training_settings']['num_iterations'],
                grpo_epochs=self.config['training_settings']['grpo_epochs'],
                gamma=self.config['training_settings']['gamma'],
                kl_beta=self.config['training_settings']['kl_beta'],
                clip_epsilon=self.config['training_settings']['clip_epsilon'],
                entropy_coeff=self.config['training_settings']['entropy_coeff'],
                max_grad_norm=self.config['training_settings']['max_grad_norm'],
                max_new_tokens=self.config['generation_settings']['vlm_generation']['max_new_tokens'],
                temperature=self.config['generation_settings']['vlm_generation']['temperature'],
                device=self.config['system_settings']['device']
            )
            logger.info("✅ GRPO Configuration created")
            
            # --- GRPO Trainer Initialization ---
            logger.info("🎯 Initializing GRPO Trainer...")
            if GRPOTrainer is None:
                raise ImportError("GRPOTrainer not available. Please install required dependencies.")
            
            self.grpo_trainer: GRPOTrainer = GRPOTrainer(
                vlm_model=self.vlm_policy,
                sd_generator=self.sd_generator,
                clip_reward=self.clip_reward,
                config=self.grpo_config
            )
            logger.info("✅ GRPO Trainer initialized")
            
            # --- Data Loader Initialization ---
            logger.info("📊 Initializing Data Loader...")
            if PromptDataLoader is None:
                logger.warning("⚠️ DataLoader not available. Using default prompts.")
                self.data_loader = None
            else:
                self.data_loader: PromptDataLoader = PromptDataLoader(
                    train_file=self.config['data_settings']['train_prompts_file'],
                    val_file=self.config['data_settings']['val_prompts_file'],
                    batch_size=self.config['training_settings']['group_size']
                )
                logger.info("✅ Data Loader initialized")
            
            # --- Validator Initialization ---
            logger.info("🔍 Initializing Validator...")
            if ValidationEvaluator is None:
                logger.warning("⚠️ Validator not available. Skipping validation.")
                self.validator = None
            else:
                self.validator: ValidationEvaluator = ValidationEvaluator(
                    vlm_model=self.vlm_policy,
                    sd_generator=self.sd_generator,
                    clip_calculator=self.clip_reward,
                    config=self.config
                )
                logger.info("✅ Validator initialized")
            
            # --- Wandb Logger Initialization ---
            logger.info("📈 Initializing Wandb Logger...")
            if WandbLogger is None:
                logger.warning("⚠️ WandbLogger not available. Skipping wandb logging.")
                self.wandb_logger = None
            else:
                self.wandb_logger: WandbLogger = WandbLogger(
                    project_name=self.config['wandb_settings']['project_name'],
                    config=self.config
                )
                logger.info("✅ Wandb Logger initialized")
            
            logger.info("🎉 All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    def run_training(self):
        """
        메인 GRPO 학습 루프 실행 - 제공된 코드 형식을 따름
        
        # --- GRPO Training Loop ---
        for iteration in range(NUM_ITERATIONS_GRPO):
            # --- 1. Collect Group of Trajectories (Rollout Phase) ---
            # --- 2. Calculate Group Relative Advantages ---
            # --- 3. Perform GRPO Update ---
            # --- Logging ---
        """
        logger.info("🚀 Starting GRPO Training Loop...")
        
        if not all([self.vlm_policy, self.sd_generator, self.clip_reward, self.grpo_trainer]):
            logger.error("❌ Components not initialized. Call initialize_components() first.")
            return
        
        # --- Training Configuration ---
        NUM_ITERATIONS_GRPO: int = self.grpo_config.num_iterations
        GROUP_SIZE: int = self.grpo_config.group_size
        INTERV_PRINT: int = max(1, NUM_ITERATIONS_GRPO // 10)
        
        # --- Get Training Prompts ---
        if self.data_loader is not None:
            train_prompts = self.data_loader.get_train_prompts()
        else:
            # Default prompts if data loader not available
            train_prompts = [
                "a beautiful sunset over mountains",
                "a cat sitting in a garden",
                "abstract art with vibrant colors",
                "a futuristic city skyline",
                "a peaceful forest scene"
            ]
        
        logger.info(f"📝 Training with {len(train_prompts)} prompts")
        logger.info(f"🔄 Starting {NUM_ITERATIONS_GRPO} iterations with group size {GROUP_SIZE}")
        
        # --- GRPO Training Loop ---
        start_time = time.time()
        
        for iteration in range(NUM_ITERATIONS_GRPO):
            iteration_start_time = time.time()
            
            logger.info(f"\n{'='*50}")
            logger.info(f"🔄 Iteration {iteration+1}/{NUM_ITERATIONS_GRPO}")
            logger.info(f"{'='*50}")
            
            try:
                # --- 1. Sample Group Prompts ---
                if len(train_prompts) >= GROUP_SIZE:
                    # 랜덤 샘플링
                    import random
                    group_prompts = random.sample(train_prompts, GROUP_SIZE)
                else:
                    # 반복 샘플링
                    group_prompts = (train_prompts * ((GROUP_SIZE // len(train_prompts)) + 1))[:GROUP_SIZE]
                
                logger.info(f"📝 Selected {len(group_prompts)} prompts for this iteration")
                
                # --- 2. Perform GRPO Training Iteration ---
                iteration_metrics = self.grpo_trainer.train_iteration(group_prompts)
                
                # --- 3. Update Training Statistics ---
                self.training_stats['current_iteration'] = iteration + 1
                self.training_stats['total_iterations'] = iteration + 1
                
                # Store metrics for plotting/logging
                self.iteration_rewards.append(iteration_metrics['avg_reward'])
                self.iteration_policy_losses.append(iteration_metrics['policy_loss'])
                self.iteration_entropies.append(iteration_metrics['entropy'])
                self.iteration_kl_divs.append(iteration_metrics['kl_div'])
                
                # --- 4. Check for Best Model ---
                if iteration_metrics['avg_reward'] > self.training_stats['best_reward']:
                    self.training_stats['best_reward'] = iteration_metrics['avg_reward']
                    self._save_best_model(iteration + 1)
                
                # --- 5. Logging ---
                iteration_time = time.time() - iteration_start_time
                self.training_stats['total_time'] += iteration_time
                
                if (iteration + 1) % INTERV_PRINT == 0 or iteration == NUM_ITERATIONS_GRPO - 1:
                    logger.info(f"\n📊 Iteration {iteration+1}/{NUM_ITERATIONS_GRPO} Summary:")
                    logger.info(f"  Avg Reward (Group): {iteration_metrics['avg_reward']:.4f}")
                    logger.info(f"  Policy Loss: {iteration_metrics['policy_loss']:.4f}")
                    logger.info(f"  Entropy: {iteration_metrics['entropy']:.4f}")
                    logger.info(f"  KL Divergence: {iteration_metrics['kl_div']:.4f}")
                    logger.info(f"  Iteration Time: {iteration_time:.2f}s")
                    logger.info(f"  Best Reward So Far: {self.training_stats['best_reward']:.4f}")
                
                # --- 6. Wandb Logging ---
                if self.wandb_logger is not None:
                    self.wandb_logger.log_metrics({
                        'iteration': iteration + 1,
                        'avg_reward': iteration_metrics['avg_reward'],
                        'policy_loss': iteration_metrics['policy_loss'],
                        'entropy': iteration_metrics['entropy'],
                        'kl_divergence': iteration_metrics['kl_div'],
                        'iteration_time': iteration_time,
                        'best_reward': self.training_stats['best_reward']
                    })
                
                # --- 7. Validation ---
                if self.validator is not None and (iteration + 1) % (INTERV_PRINT * 2) == 0:
                    self._run_validation(iteration + 1)
                
                # --- 8. Checkpoint Saving ---
                if (iteration + 1) % (INTERV_PRINT * 2) == 0:
                    self._save_checkpoint(iteration + 1)
                
            except Exception as e:
                logger.error(f"❌ Error in iteration {iteration+1}: {e}")
                continue
        
        # --- Training Complete ---
        total_time = time.time() - start_time
        self.training_stats['total_time'] = total_time
        
        logger.info(f"\n🎉 GRPO Training Loop Finished!")
        logger.info(f"📊 Final Statistics:")
        logger.info(f"  Total Iterations: {NUM_ITERATIONS_GRPO}")
        logger.info(f"  Total Time: {total_time:.2f}s")
        logger.info(f"  Average Time per Iteration: {total_time/NUM_ITERATIONS_GRPO:.2f}s")
        logger.info(f"  Best Reward Achieved: {self.training_stats['best_reward']:.4f}")
        
        # --- Save Final Results ---
        self._save_final_results()
        
        if self.wandb_logger is not None:
            self.wandb_logger.finish()
    
    def _run_validation(self, iteration: int):
        """검증 실행"""
        try:
            logger.info(f"🔍 Running validation at iteration {iteration}...")
            if self.validator is None:
                logger.warning("⚠️ Validator not available, skipping validation")
                return
            
            val_results = self.validator.evaluate()
            logger.info(f"✅ Validation complete - Score: {val_results.get('avg_score', 0.0):.4f}")
            
            if self.wandb_logger is not None:
                self.wandb_logger.log_metrics({
                    'val_score': val_results.get('avg_score', 0.0),
                    'val_iteration': iteration
                })
                
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
    
    def _save_checkpoint(self, iteration: int):
        """체크포인트 저장"""
        try:
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"grpo_checkpoint_iter_{iteration}.pt"
            self.grpo_trainer.save_checkpoint(str(checkpoint_path))
            
            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
    
    def _save_best_model(self, iteration: int):
        """최고 성능 모델 저장"""
        try:
            best_model_dir = Path("best_models")
            best_model_dir.mkdir(exist_ok=True)
            
            best_model_path = best_model_dir / f"best_grpo_model_iter_{iteration}.pt"
            self.grpo_trainer.save_checkpoint(str(best_model_path))
            
            logger.info(f"🏆 Best model saved: {best_model_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save best model: {e}")
    
    def _save_final_results(self):
        """최종 결과 저장"""
        try:
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            # 학습 통계 저장
            import json
            stats_path = results_dir / "training_stats.json"
            with open(stats_path, 'w') as f:
                json.dump({
                    **self.training_stats,
                    'iteration_rewards': self.iteration_rewards,
                    'iteration_policy_losses': self.iteration_policy_losses,
                    'iteration_entropies': self.iteration_entropies,
                    'iteration_kl_divs': self.iteration_kl_divs
                }, f, indent=2)
            
            logger.info(f"📊 Training statistics saved: {stats_path}")
            
            # 최종 모델 저장
            final_model_path = results_dir / "final_grpo_model.pt"
            self.grpo_trainer.save_checkpoint(str(final_model_path))
            
            logger.info(f"💾 Final model saved: {final_model_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save final results: {e}")

def main():
    """메인 실행 함수"""
    try:
        # VLM GRPO System 초기화
        system = VLMGRPOSystem()
        
        # 컴포넌트 초기화
        system.initialize_components()
        
        # 학습 실행
        system.run_training()
        
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 