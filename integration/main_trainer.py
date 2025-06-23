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

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

# 각 모듈 임포트
try:
    from models.vlm_wrapper import VLMWrapper
    from models.sd_generator import SD3Generator  
    from models.clip_reward import CLIPRewardCalculator, MultiRewardCalculator
    from training.grpo_trainer import GRPOTrainer, GRPOConfig
    from utils.data_loader import PromptDataLoader
    from evaluation.validator import ValidationEvaluator
    from integration.wandb_logger import WandbLogger
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    print("일부 모듈을 찾을 수 없습니다. 실제 실행 시에는 모든 의존성이 설치되어 있어야 합니다.")

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
    
    def __init__(self, config_path: Optional[str] = None):
        """
        VLM GRPO System 초기화
        
        Args:
            config_path (str, optional): 설정 파일 경로
        """
        # 설정 로딩
        self.config = self._load_config(config_path)
        
        # 로깅 설정
        self._setup_logging()
        
        # 컴포넌트 초기화
        self.vlm = None
        self.sd_generator = None
        self.clip_calculator = None
        self.multi_reward_calculator = None
        self.grpo_trainer = None
        self.data_loader = None
        self.validator = None
        self.wandb_logger = None
        
        # 학습 상태
        self.training_stats = {
            'iteration': 0,
            'total_time': 0,
            'best_reward': -float('inf'),
            'best_model_path': None
        }
        
        logger.info("🚀 VLM GRPO System initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """
        설정 파일 로딩 또는 기본 설정 생성
        
        Args:
            config_path (str, optional): 설정 파일 경로
            
        Returns:
            Dict: 시스템 설정
        """
        default_config = {
            # 모델 설정
            "vlm_model": "microsoft/DialoGPT-medium",
            "sd_model": "runwayml/stable-diffusion-v1-5", 
            "clip_model": "openai/clip-vit-base-patch32",
            
            # 학습 설정
            "learning_rate": 1e-5,
            "group_size": 4,
            "num_iterations": 50,
            "grpo_epochs": 2,
            "validation_interval": 5,
            
            # 데이터 설정
            "train_data_path": "train_prompts.jsonl",
            "val_data_path": "val_prompts.jsonl",
            
            # 출력 설정
            "output_dir": "vlm_grpo_results",
            "checkpoint_interval": 10,
            "save_images": True,
            
            # Wandb 설정
            "use_wandb": True,
            "wandb_project": "vlm-grpo-training",
            "wandb_entity": None,
            
            # 디바이스 설정
            "device": "auto"
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"📥 Config loaded from {config_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load config: {e}, using defaults")
        
        return default_config
    
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
        모든 컴포넌트 초기화
        
        이 메서드는 시스템의 모든 구성 요소를 순서대로 초기화합니다:
        1. VLM (프롬프트 개선)
        2. SD3 Generator (이미지 생성)
        3. CLIP Reward Calculator (보상 계산)
        4. GRPO Trainer (강화학습)
        5. Data Loader (데이터 관리)
        6. Validator (검증)
        7. Wandb Logger (실험 추적)
        """
        try:
            logger.info("🔧 Initializing components...")
            
            # 1. VLM 초기화
            logger.info("📝 Initializing VLM...")
            self.vlm = VLMWrapper(
                model_name=self.config["vlm_model"],
                device=self.config["device"]
            )
            
            # 2. SD3 Generator 초기화
            logger.info("🎨 Initializing SD3 Generator...")
            self.sd_generator = SD3Generator(
                model_name=self.config["sd_model"],
                device=self.config["device"]
            )
            
            # 3. CLIP Reward Calculator 초기화
            logger.info("🏆 Initializing CLIP Reward Calculator...")
            self.clip_calculator = CLIPRewardCalculator(
                model_name=self.config["clip_model"],
                device=self.config["device"]
            )
            
            # 4. Multi Reward Calculator 초기화
            self.multi_reward_calculator = MultiRewardCalculator(
                self.clip_calculator
            )
            
            # 5. GRPO Trainer 초기화
            logger.info("🎯 Initializing GRPO Trainer...")
            grpo_config = GRPOConfig(
                learning_rate=self.config["learning_rate"],
                group_size=self.config["group_size"],
                num_iterations=self.config["num_iterations"],
                grpo_epochs=self.config["grpo_epochs"],
                device=self.config["device"]
            )
            self.grpo_trainer = GRPOTrainer(self.vlm, grpo_config)
            
            # 6. Data Loader 초기화
            logger.info("📊 Initializing Data Loader...")
            self.data_loader = PromptDataLoader(
                train_data_path=self.config["train_data_path"],
                val_data_path=self.config["val_data_path"]
            )
            
            # 7. Validator 초기화
            logger.info("✅ Initializing Validator...")
            self.validator = ValidationEvaluator(
                vlm=self.vlm,
                sd_generator=self.sd_generator,
                clip_calculator=self.clip_calculator
            )
            
            # 8. Wandb Logger 초기화 (선택적)
            if self.config["use_wandb"]:
                logger.info("📈 Initializing Wandb Logger...")
                self.wandb_logger = WandbLogger(
                    project=self.config["wandb_project"],
                    entity=self.config["wandb_entity"],
                    config=self.config
                )
            
            # 출력 디렉토리 생성
            os.makedirs(self.config["output_dir"], exist_ok=True)
            
            logger.info("✅ All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    def run_training(self):
        """
        메인 학습 루프 실행
        
        이 메서드는 GRPO 학습의 전체 과정을 실행합니다:
        1. 학습 데이터 배치 생성
        2. VLM으로 프롬프트 개선
        3. SD3로 이미지 생성
        4. CLIP으로 보상 계산
        5. GRPO 정책 업데이트
        6. 주기적 검증 및 저장
        """
        logger.info("🚀 Starting VLM GRPO training...")
        start_time = time.time()
        
        try:
            for iteration in range(self.config["num_iterations"]):
                iteration_start = time.time()
                
                logger.info(f"🔄 Iteration {iteration + 1}/{self.config['num_iterations']}")
                
                # 1. 학습 배치 생성
                batch_prompts = self.data_loader.get_training_batch(
                    batch_size=self.config["group_size"]
                )
                
                if not batch_prompts:
                    logger.warning("⚠️ No training data available, skipping iteration")
                    continue
                
                # 2. 그룹 데이터 수집 (VLM + SD3 + CLIP)
                group_data = self._collect_training_data(batch_prompts)
                
                # 3. GRPO 업데이트
                training_metrics = self.grpo_trainer.grpo_update(group_data)
                
                # 4. 메트릭 로깅
                iteration_time = time.time() - iteration_start
                self._log_training_metrics(iteration + 1, training_metrics, iteration_time)
                
                # 5. 주기적 검증
                if (iteration + 1) % self.config["validation_interval"] == 0:
                    self._run_validation(iteration + 1)
                
                # 6. 체크포인트 저장
                if (iteration + 1) % self.config["checkpoint_interval"] == 0:
                    self._save_checkpoint(iteration + 1)
                
                # 7. 최고 성능 모델 저장
                avg_reward = training_metrics.get('avg_reward', 0)
                if avg_reward > self.training_stats['best_reward']:
                    self.training_stats['best_reward'] = avg_reward
                    self._save_best_model(iteration + 1)
            
            # 학습 완료
            total_time = time.time() - start_time
            self.training_stats['total_time'] = total_time
            
            logger.info(f"✅ Training completed! Total time: {total_time:.2f}s")
            self._save_final_results()
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
        finally:
            # Wandb 세션 종료
            if self.wandb_logger:
                self.wandb_logger.finish()
    
    def _collect_training_data(self, prompts: List[str]) -> Dict[str, Any]:
        """
        학습 데이터 수집: VLM + SD3 + CLIP 파이프라인
        
        Args:
            prompts (List[str]): 입력 프롬프트들
            
        Returns:
            Dict[str, Any]: 수집된 학습 데이터
        """
        logger.debug(f"📊 Collecting training data for {len(prompts)} prompts")
        
        group_data = {
            'prompts': prompts,
            'enhanced_prompts': [],
            'images': [],
            'rewards': [],
            'comprehensive_rewards': []
        }
        
        for prompt in prompts:
            try:
                # 1. VLM으로 프롬프트 개선
                enhanced_prompt = self.vlm.enhance_prompt(prompt)
                
                # 2. SD3로 이미지 생성
                image = self.sd_generator.generate_image(enhanced_prompt)
                
                # 3. 종합적 보상 계산
                rewards = self.multi_reward_calculator.calculate_comprehensive_reward(
                    image, prompt, enhanced_prompt
                )
                
                # 데이터 저장
                group_data['enhanced_prompts'].append(enhanced_prompt)
                group_data['images'].append(image)
                group_data['rewards'].append(rewards['final_reward'])
                group_data['comprehensive_rewards'].append(rewards)
                
                logger.debug(f"✅ Processed: '{prompt}' → reward: {rewards['final_reward']:.4f}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to process prompt '{prompt}': {e}")
                # 실패 시 기본값 사용
                group_data['enhanced_prompts'].append(prompt)
                group_data['images'].append(None)
                group_data['rewards'].append(0.0)
                group_data['comprehensive_rewards'].append({'final_reward': 0.0})
        
        return group_data
    
    def _log_training_metrics(self, iteration: int, metrics: Dict, iteration_time: float):
        """
        학습 메트릭 로깅
        
        Args:
            iteration (int): 현재 반복 횟수
            metrics (Dict): 학습 메트릭
            iteration_time (float): 반복 시간
        """
        # 기본 로깅
        logger.info(f"📊 Iteration {iteration} metrics:")
        logger.info(f"  - Policy Loss: {metrics.get('policy_loss', 0):.6f}")
        logger.info(f"  - KL Divergence: {metrics.get('kl_div', 0):.6f}")
        logger.info(f"  - Entropy: {metrics.get('entropy', 0):.6f}")
        logger.info(f"  - Average Reward: {metrics.get('avg_reward', 0):.4f}")
        logger.info(f"  - Iteration Time: {iteration_time:.2f}s")
        
        # Wandb 로깅
        if self.wandb_logger:
            wandb_metrics = {
                'iteration': iteration,
                'policy_loss': metrics.get('policy_loss', 0),
                'kl_divergence': metrics.get('kl_div', 0),
                'entropy': metrics.get('entropy', 0),
                'average_reward': metrics.get('avg_reward', 0),
                'iteration_time': iteration_time,
                'total_time': time.time() - self.start_time if hasattr(self, 'start_time') else 0
            }
            self.wandb_logger.log_training_metrics(wandb_metrics)
    
    def _run_validation(self, iteration: int):
        """
        검증 실행
        
        Args:
            iteration (int): 현재 반복 횟수
        """
        logger.info(f"🔍 Running validation at iteration {iteration}")
        
        try:
            # 검증 데이터 가져오기
            val_data = self.data_loader.get_validation_data()
            
            if not val_data:
                logger.warning("⚠️ No validation data available")
                return
            
            # 검증 실행
            val_results = self.validator.evaluate_batch(val_data[:10])  # 처음 10개만
            
            # 결과 로깅
            logger.info(f"✅ Validation results:")
            logger.info(f"  - Success Rate: {val_results.get('success_rate', 0):.2%}")
            logger.info(f"  - Average CLIP Score: {val_results.get('avg_clip_score', 0):.4f}")
            logger.info(f"  - Quality Score: {val_results.get('quality_score', 0):.4f}")
            
            # Wandb 로깅
            if self.wandb_logger:
                self.wandb_logger.log_validation_results(val_results)
            
            # 검증 결과 저장
            val_save_path = f"{self.config['output_dir']}/validation_iter_{iteration}.json"
            with open(val_save_path, 'w', encoding='utf-8') as f:
                json.dump(val_results, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
    
    def _save_checkpoint(self, iteration: int):
        """체크포인트 저장"""
        checkpoint_path = f"{self.config['output_dir']}/checkpoint_iter_{iteration}.pt"
        self.grpo_trainer.save_checkpoint(checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def _save_best_model(self, iteration: int):
        """최고 성능 모델 저장"""
        best_model_path = f"{self.config['output_dir']}/best_model.pt"
        self.grpo_trainer.save_checkpoint(best_model_path)
        self.training_stats['best_model_path'] = best_model_path
        logger.info(f"🏆 Best model saved: {best_model_path} (iteration {iteration})")
    
    def _save_final_results(self):
        """최종 결과 저장"""
        results = {
            'config': self.config,
            'training_stats': self.training_stats,
            'final_metrics': self.grpo_trainer.get_training_stats()
        }
        
        results_path = f"{self.config['output_dir']}/final_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 Final results saved: {results_path}")


def main():
    """메인 실행 함수"""
    print("🚀 VLM GRPO System Starting...")
    print("=" * 50)
    
    try:
        # 시스템 초기화
        system = VLMGRPOSystem()
        
        # 컴포넌트 초기화
        system.initialize_components()
        
        # 학습 실행
        system.run_training()
        
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main() 