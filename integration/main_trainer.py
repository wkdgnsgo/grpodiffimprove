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
    from utils.data_loader import DataLoader
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
            'best_reward': float('-inf'),
            'total_iterations': 0,
            'total_time': 0.0
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
            if VLMWrapper is None:
                raise ImportError("VLMWrapper not available. Please install required dependencies.")
            self.vlm = VLMWrapper(
                config_path=self.config_path,
                device=self.config['system_settings']['device'],
                max_new_tokens=self.config['generation_settings']['vlm_generation']['max_new_tokens'],
                temperature=self.config['generation_settings']['vlm_generation']['temperature'],
                top_p=self.config['generation_settings']['vlm_generation']['top_p']
            )
            
            # 2. SD3 Generator 초기화
            logger.info("🎨 Initializing SD3 Generator...")
            if SD3Generator is None:
                raise ImportError("SD3Generator not available. Please install required dependencies.")
            self.sd_generator = SD3Generator(
                config_path=self.config_path,
                device=self.config['system_settings']['device'],
                height=self.config['generation_settings']['sd_generation']['height'],
                width=self.config['generation_settings']['sd_generation']['width'],
                num_inference_steps=self.config['generation_settings']['sd_generation']['num_inference_steps'],
                guidance_scale=self.config['generation_settings']['sd_generation']['guidance_scale']
            )
            
            # 3. CLIP Reward Calculator 초기화
            logger.info("🏆 Initializing CLIP Reward Calculator...")
            if CLIPRewardCalculator is None:
                raise ImportError("CLIPRewardCalculator not available. Please install required dependencies.")
            self.clip_calculator = CLIPRewardCalculator(
                model_name=self.config['model_settings']['clip_model'],
                device=self.config['system_settings']['device'],
                reward_weights=self.config['reward_settings']['reward_weights']
            )
            
            # 4. Multi Reward Calculator 초기화
            if MultiRewardCalculator is None:
                raise ImportError("MultiRewardCalculator not available. Please install required dependencies.")
            self.multi_reward_calculator = MultiRewardCalculator(
                self.clip_calculator
            )
            
            # 5. GRPO Trainer 초기화
            logger.info("🎯 Initializing GRPO Trainer...")
            if GRPOTrainer is None or GRPOConfig is None:
                raise ImportError("GRPO modules not available. Please install required dependencies.")
            grpo_config = GRPOConfig(
                learning_rate=self.config['training_settings']['learning_rate'],
                group_size=self.config['training_settings']['group_size'],
                num_iterations=self.config['training_settings']['num_iterations'],
                grpo_epochs=self.config['training_settings']['grpo_epochs'],
                gamma=self.config['training_settings']['grpo_parameters']['gamma'],
                kl_beta=self.config['training_settings']['grpo_parameters']['kl_beta'],
                clip_epsilon=self.config['training_settings']['grpo_parameters']['clip_epsilon'],
                entropy_coeff=self.config['training_settings']['grpo_parameters']['entropy_coeff'],
                max_grad_norm=self.config['training_settings']['grpo_parameters']['max_grad_norm'],
                epsilon_std=self.config['training_settings']['grpo_parameters']['epsilon_std'],
                max_new_tokens=self.config['generation_settings']['vlm_generation']['max_new_tokens'],
                vocab_size=50000,  # GPT-2 기본 vocab 크기
                max_sequence_length=100,  # 최대 시퀀스 길이
                temperature=self.config['generation_settings']['vlm_generation']['temperature'],
                device=self.config['system_settings']['device']
            )
            self.grpo_trainer = GRPOTrainer(
                vlm_model=self.vlm,
                sd_generator=self.sd_generator,
                clip_reward=self.clip_calculator,
                config=grpo_config
            )
            
            # 6. Data Loader 초기화
            logger.info("📊 Initializing Data Loader...")
            try:
                self.data_loader = DataLoader(
                    train_data_path=self.config['data_settings']['train_data_path'],
                    val_data_path=self.config['data_settings']['val_data_path'],
                    batch_shuffle=self.config['data_settings']['batch_shuffle']
                )
            except Exception as e:
                logger.error(f"❌ Data Loader initialization failed: {e}")
                self.data_loader = None
            
            # 7. Validator 초기화
            logger.info("✅ Initializing Validator...")
            if ValidationEvaluator is None:
                raise ImportError("ValidationEvaluator not available. Please install required dependencies.")
            self.validator = ValidationEvaluator(
                vlm=self.vlm,
                sd_generator=self.sd_generator,
                clip_calculator=self.clip_calculator
            )
            
            # 8. Wandb Logger 초기화 (선택적)
            if self.config.get("wandb_settings", {}).get("use_wandb", False):
                logger.info("📈 Initializing Wandb Logger...")
                if WandbLogger is None:
                    logger.warning("⚠️ WandbLogger not available, skipping wandb initialization")
                else:
                    self.wandb_logger = WandbLogger(
                        project=self.config.get("wandb_settings", {}).get("project", "vlm-grpo"),
                        entity=self.config.get("wandb_settings", {}).get("entity", None),
                        config=self.config
                    )
            
            # 출력 디렉토리 생성
            os.makedirs(self.config["output_settings"]["output_dir"], exist_ok=True)
            
            # 학습 통계 초기화
            self.training_stats = {
                'best_reward': float('-inf'),
                'total_iterations': 0,
                'total_time': 0.0
            }
            
            logger.info("✅ All components initialized successfully")
            
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
        self.start_time = time.time()
        
        try:
            for iteration in range(self.config["training_settings"]["num_iterations"]):
                iteration_start = time.time()
                
                logger.info(f"🔄 Iteration {iteration + 1}/{self.config['training_settings']['num_iterations']}")
                
                # 1. 학습 배치 생성
                if self.data_loader is None:
                    logger.error("❌ Data loader not initialized")
                    break
                    
                batch_prompts = self.data_loader.get_training_batch(
                    batch_size=self.config["training_settings"]["group_size"]
                )
                
                if not batch_prompts:
                    logger.warning("⚠️ No training data available, skipping iteration")
                    continue
                
                # 2. GRPO 그룹 데이터 수집 (토큰별 순차 생성)
                if self.grpo_trainer is None:
                    logger.error("❌ GRPO trainer not initialized")
                    break
                
                # GRPOTrainer의 collect_group_data 메서드 사용
                group_data = self.grpo_trainer.collect_group_data(batch_prompts)
                
                # 3. GRPO 업데이트
                training_metrics = self.grpo_trainer.grpo_update(group_data)
                
                # 4. 메트릭 로깅
                iteration_time = time.time() - iteration_start
                self._log_training_metrics(iteration + 1, training_metrics, iteration_time)
                
                # 5. 주기적 검증
                if (iteration + 1) % self.config["training_settings"]["validation_interval"] == 0:
                    self._run_validation(iteration + 1)
                
                # 6. 체크포인트 저장
                if (iteration + 1) % self.config["training_settings"]["checkpoint_interval"] == 0:
                    self._save_checkpoint(iteration + 1)
                
                # 7. 최고 성능 모델 저장
                avg_reward = training_metrics.get('avg_reward', 0)
                if avg_reward > self.training_stats['best_reward']:
                    self.training_stats['best_reward'] = avg_reward
                    self._save_best_model(iteration + 1)
            
            # 학습 완료
            total_time = time.time() - self.start_time
            self.training_stats['total_time'] = total_time
            
            logger.info(f"✅ Training completed! Total time: {total_time:.2f}s")
            self._save_final_results()
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
        finally:
            # Wandb 세션 종료
            if hasattr(self, 'wandb_logger') and self.wandb_logger:
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
                if self.vlm is None:
                    logger.warning("⚠️ VLM not initialized, using original prompt")
                    enhanced_prompt = prompt
                else:
                    enhanced_prompt = self.vlm.enhance_prompt(prompt)
                
                # 2. SD3로 이미지 생성
                if self.sd_generator is None:
                    logger.warning("⚠️ SD generator not initialized, skipping image generation")
                    image = None
                else:
                    image = self.sd_generator.generate_image(enhanced_prompt)
                
                # 3. 종합적 보상 계산
                if self.multi_reward_calculator is None:
                    logger.warning("⚠️ Multi reward calculator not initialized, using default reward")
                    rewards = {'final_reward': 0.0}
                else:
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
            if self.data_loader is None:
                logger.warning("⚠️ Data loader not initialized, skipping validation")
                return
                
            val_data = self.data_loader.get_validation_data()
            
            if not val_data:
                logger.warning("⚠️ No validation data available")
                return
            
            # 검증 실행 (이미지 저장 포함)
            if self.validator is None:
                logger.warning("⚠️ Validator not initialized, skipping validation")
                return
            
            # 이미지 저장 설정 확인
            save_images = self.config.get("output_settings", {}).get("save_images", True)
            output_dir = self.config.get("output_settings", {}).get("output_dir", "vlm_grpo_results")
            
            val_results = self.validator.evaluate_batch(
                val_data[:10],  # 처음 10개만
                save_images=save_images,
                output_dir=output_dir,
                iteration=iteration
            )
            
            # 결과 로깅
            logger.info(f"📊 Validation Results (Iteration {iteration}):")
            for metric, value in val_results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  - {metric}: {value:.4f}")
            
            # 이미지 저장 결과 로깅
            if save_images and val_results.get('saved_images'):
                saved_count = len(val_results['saved_images'])
                logger.info(f"💾 Saved {saved_count} validation images")
                
                # 저장된 이미지 정보 로깅 (처음 3개만)
                for i, img_info in enumerate(val_results['saved_images'][:3]):
                    logger.info(f"  📸 Image {i+1}:")
                    logger.info(f"    Original: '{img_info['prompt'][:30]}...'")
                    logger.info(f"    Enhanced: '{img_info['enhanced_prompt'][:50]}...'")
                    logger.info(f"    Enhanced Path: {img_info['image_path']}")
                    if 'saved_original_path' in img_info:
                        logger.info(f"    Original Path: {img_info['saved_original_path']}")
                    if 'saved_prompts_path' in img_info:
                        logger.info(f"    Prompts File: {img_info['saved_prompts_path']}")
                    logger.info(f"    CLIP Score: {img_info['clip_score']:.3f}")
                    logger.info("")
            
            # Wandb 로깅
            if hasattr(self, 'wandb_logger') and self.wandb_logger:
                self.wandb_logger.log_validation_results(val_results)
                
                # 이미지도 wandb에 업로드 (가능한 경우)
                if save_images and val_results.get('saved_images'):
                    try:
                        from PIL import Image
                        images_for_wandb = []
                        captions_for_wandb = []
                        
                        for img_info in val_results['saved_images'][:5]:  # 처음 5개만
                            try:
                                img_path = img_info['image_path']
                                if os.path.exists(img_path):
                                    pil_image = Image.open(img_path)
                                    images_for_wandb.append(pil_image)
                                    caption = f"Iter {iteration}: {img_info['prompt'][:30]}... (CLIP: {img_info['clip_score']:.3f})"
                                    captions_for_wandb.append(caption)
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to load image for wandb: {e}")
                        
                        if images_for_wandb:
                            self.wandb_logger.log_images(images_for_wandb, captions_for_wandb, step=iteration)
                            logger.info(f"📈 Uploaded {len(images_for_wandb)} images to wandb")
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to upload images to wandb: {e}")
                
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
    
    def _save_checkpoint(self, iteration: int):
        """체크포인트 저장"""
        if self.grpo_trainer is None:
            logger.warning("⚠️ GRPO trainer not initialized, skipping checkpoint save")
            return
            
        try:
            checkpoint_path = f"{self.config['output_settings']['output_dir']}/checkpoint_iter_{iteration}.pt"
            self.grpo_trainer.save_checkpoint(checkpoint_path)
            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
    
    def _save_best_model(self, iteration: int):
        """최고 성능 모델 저장"""
        if self.grpo_trainer is None:
            logger.warning("⚠️ GRPO trainer not initialized, skipping best model save")
            return
            
        try:
            best_model_path = f"{self.config['output_settings']['output_dir']}/best_model.pt"
            self.grpo_trainer.save_checkpoint(best_model_path)
            logger.info(f"🏆 Best model saved: {best_model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save best model: {e}")
    
    def _save_final_results(self):
        """최종 결과 저장"""
        try:
            results_path = f"{self.config['output_settings']['output_dir']}/final_results.json"
            
            # 학습 통계 수집
            final_stats = self.training_stats.copy()
            
            # GRPO 통계 추가 (있다면)
            if self.grpo_trainer is not None and hasattr(self.grpo_trainer, 'get_training_stats'):
                grpo_stats = self.grpo_trainer.get_training_stats()
                final_stats.update(grpo_stats)
            
            # 결과 저장
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(final_stats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📋 Final results saved: {results_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save final results: {e}")


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