#!/usr/bin/env python3
"""
QWEN 통합 GRPO 트레이너
QWEN 모델의 enhance_prompt 기능과 GRPO를 통합하여 프롬프트 개선
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import math
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from qwen import QWENModel, QWENGRPOConfig

logger = logging.getLogger(__name__)

class QWENGRPOEnvironment:
    """QWEN GRPO 통합 환경 (Accelerate 지원)"""
    
    def __init__(self, qwen_model: QWENModel, reward_model, sd_pipeline, config: QWENGRPOConfig):
        self.qwen_model = qwen_model
        self.reward_model = reward_model
        self.sd_pipeline = sd_pipeline
        self.config = config
        
        # 멀티 프로세스 환경에서 메인 프로세스 여부 확인
        self.is_main_process = reward_model is not None and sd_pipeline is not None
        
        # GPU 디바이스 설정 (Accelerate 멀티 GPU 환경)
        self.qwen_device = "auto"         # Accelerate가 관리
        self.sd_device = "cuda:6"         # SD3 (GPU 6번)
        self.reward_device = "cuda:5"     # CLIP Reward (GPU 5번)
        self.ref_device = "cuda:5"        # Reference model (GPU 5번)
        
        self.current_user_prompt = ""
        self.current_enhanced_prompt = ""
        self.episode_count = 0
        
        # 로깅 디렉토리 설정
        if config.save_images:
            self.base_log_dir = config.log_dir
            os.makedirs(self.base_log_dir, exist_ok=True)
            logger.info(f"결과 저장 디렉토리: {self.base_log_dir}")
    
    def reset(self, user_prompt: str) -> Dict:
        """환경 리셋"""
        self.current_user_prompt = user_prompt
        self.current_enhanced_prompt = ""
        self.episode_count += 1
        
        # 에피소드 디렉토리 생성
        if self.config.save_images:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c for c in user_prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt[:30] + "..." if len(safe_prompt) > 30 else safe_prompt
            
            self.episode_dir = os.path.join(
                self.base_log_dir,
                f"episode_{self.episode_count:03d}_{timestamp}_{safe_prompt.replace(' ', '_')}"
            )
            os.makedirs(self.episode_dir, exist_ok=True)
        
        logger.info(f"🔄 환경 리셋: '{user_prompt}'")
        
        # 초기 상태 반환 (사용자 프롬프트)
        return {
            'user_prompt': self.current_user_prompt,
            'enhanced_prompt': '',
            'episode': self.episode_count
        }
    
    def step(self, enhanced_prompt: str) -> Tuple[Dict, float, bool, Dict]:
        """환경 스텝 - 단순화된 버전 (배치 처리는 trainer에서 담당)"""
        # 기본 리워드 반환 (실제 처리는 trainer의 배치 메서드에서 수행)
        reward = 0.3  # 기본 리워드
        
        next_state = {
            'user_prompt': self.current_user_prompt,
            'enhanced_prompt': enhanced_prompt,
            'episode': self.episode_count
        }
        
        info = {
            'original_prompt': self.current_user_prompt,
            'enhanced_prompt': enhanced_prompt,
            'original_reward': reward,
            'enhanced_reward': reward
        }
        
        # 에피소드 카운터 증가
        self.episode_count += 1
        
        return next_state, reward, True, info
    
    def _save_episode_results(self, original_image, enhanced_image, original_reward, enhanced_reward, total_reward, enhanced_prompt):
        """에피소드 결과 저장"""
        try:
            # 이미지 저장
            original_image.save(os.path.join(self.episode_dir, "original_image.png"))
            enhanced_image.save(os.path.join(self.episode_dir, "enhanced_image.png"))
            
            # 로그 파일 작성
            log_content = f"""=== QWEN GRPO Episode Results ===
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Episode: {self.episode_count}

=== Prompts ===
Original Prompt: {self.current_user_prompt}
Enhanced Prompt: {self.current_enhanced_prompt}

=== GRPO Direct Generation ===
Generated Enhanced Prompt: {enhanced_prompt}

=== Reward Components ===
Original Reward (Original→Original): {original_reward:.4f}
Enhanced Reward (Original→Enhanced): {enhanced_reward:.4f}
Total Reward: {total_reward:.4f}

=== Improvement ===
Reward Improvement: {enhanced_reward - original_reward:.4f}
Relative Improvement: {((enhanced_reward - original_reward) / max(original_reward, 0.001) * 100):.2f}%

=== Files ===
Original Image: original_image.png
Enhanced Image: enhanced_image.png
"""
            
            # 로그 파일 저장
            with open(os.path.join(self.episode_dir, "episode_log.txt"), "w", encoding="utf-8") as f:
                f.write(log_content)
            
            logger.info(f"💾 에피소드 결과 저장 완료: {self.episode_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to save episode results: {e}")
    
    def _save_error_log(self, enhanced_prompt: str, error_msg: str):
        """에러 발생 시 로그만 저장"""
        try:
            # 에러 로그 파일 작성
            error_log_content = f"""=== QWEN GRPO Error Log ===
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Episode: {self.episode_count}

=== Error Info ===
Error Message: {error_msg}
Original Prompt: {self.current_user_prompt}
Enhanced Prompt: {enhanced_prompt}

=== Note ===
Images could not be generated/saved due to the error above.
"""
            
            # 에러 로그 파일 저장
            error_log_path = os.path.join(self.episode_dir, "error_log.txt")
            with open(error_log_path, "w", encoding="utf-8") as f:
                f.write(error_log_content)
            
            logger.info(f"📝 에러 로그 저장 완료: {error_log_path}")
            
        except Exception as log_error:
            logger.error(f"❌ 에러 로그 저장도 실패: {log_error}")

class QWENGRPOTrainer:
    """QWEN 통합 GRPO 트레이너 (Accelerate 지원)"""
    
    def __init__(self, qwen_model: QWENModel, reward_model, sd_pipeline, config: QWENGRPOConfig):
        self.config = config
        self.qwen_model = qwen_model
        self.reward_model = reward_model
        self.sd_pipeline = sd_pipeline
        self.accelerator = None  # Accelerate 객체 (나중에 설정)
        
        # 멀티 프로세스 환경에서 메인 프로세스 여부 확인
        self.is_main_process = reward_model is not None and sd_pipeline is not None
        
        # QWEN 모델에 GRPO 컴포넌트가 설정되어 있는지 확인
        if not hasattr(qwen_model, 'ref_model'):
            raise ValueError("QWEN 모델에 GRPO 컴포넌트가 설정되지 않았습니다. grpo_config를 전달하여 초기화하세요.")
        
        self.env = QWENGRPOEnvironment(qwen_model, reward_model, sd_pipeline, config)
        
        # 리워드 추적을 위한 변수들
        self.episode_rewards = []
        self.episode_numbers = []
        self.running_avg_rewards = []
        
        # 플롯 저장을 위한 디렉토리 설정 (Environment와 동일한 디렉토리 사용)
        self.plot_save_dir = self.env.base_log_dir if config.save_images else config.log_dir
        os.makedirs(self.plot_save_dir, exist_ok=True)
        
        # 플롯 설정
        mplstyle.use('fast')
        plt.ion()  # Interactive mode on
        
        logger.info("🎯 QWEN GRPO 트레이너 초기화 완료")
        logger.info(f"✅ QWEN 직접 학습 방식으로 설정 (Accelerate 지원)")
        logger.info(f"📊 플롯 저장 디렉토리: {self.plot_save_dir}")
    
    def collect_rollouts(self, prompts: List[str], is_baseline: bool = False) -> List[Dict]:
        """롤아웃 수집 - 배치 이미지 생성 최적화"""
        all_experiences = []
        rollout_type = "베이스라인" if is_baseline else "학습용"
        
        for prompt_idx, user_prompt in enumerate(prompts):
            logger.info(f"\n📝 {rollout_type} 프롬프트 {prompt_idx + 1}/{len(prompts)}: '{user_prompt}'")
            
            # 배치 롤아웃 수집 (모든 프로세스의 프롬프트를 한번에 처리)
            batch_experiences = self.collect_batch_rollouts(user_prompt, is_baseline)
            all_experiences.extend(batch_experiences)
        
        logger.info(f"\n📊 수집된 {rollout_type} 경험: {len(all_experiences)}개")
        return all_experiences
    
    def collect_batch_rollouts(self, user_prompt: str, is_baseline: bool = False) -> List[Dict]:
        """배치 롤아웃 수집 - 메인 프로세스에서 배치 이미지 생성"""
        batch_experiences = []
        
        # 모든 프로세스에서 향상된 프롬프트 생성
        enhanced_prompts = []
        log_probs = []
        
        for rollout_idx in range(self.config.num_rollouts):
            logger.info(f"  🎲 롤아웃 {rollout_idx + 1}/{self.config.num_rollouts}")
            
            try:
                # QWEN GRPO로 향상된 프롬프트 생성
                if self.accelerator:
                    with self.accelerator.device:
                        enhanced_prompt, log_prob = self.qwen_model.generate_grpo_enhanced_prompt(user_prompt)
                else:
                    enhanced_prompt, log_prob = self.qwen_model.generate_grpo_enhanced_prompt(user_prompt)
                
                enhanced_prompts.append(enhanced_prompt)
                log_probs.append(log_prob)
                logger.info(f"    🎯 생성된 프롬프트: '{enhanced_prompt[:50]}...' (로그 확률: {log_prob:.4f})")
                
            except Exception as e:
                logger.error(f"    ❌ 프롬프트 생성 오류: {e}")
                continue
        
        # 모든 프로세스에서 배치 이미지 생성 및 리워드 계산
        if enhanced_prompts:
            batch_rewards = self.generate_batch_images_and_rewards(user_prompt, enhanced_prompts)
        else:
            batch_rewards = []
        
        # 각 프로세스가 독립적으로 리워드 계산하므로 분배 로직 불필요
        logger.info(f"🎯 배치 리워드 계산 완료: {len(batch_rewards)}개")
        
        # 경험 생성
        for enhanced_prompt, log_prob, reward in zip(enhanced_prompts, log_probs, batch_rewards):
            experience = {
                'user_prompt': user_prompt,
                'enhanced_prompt': enhanced_prompt,
                'log_prob': log_prob,
                'reward': reward,
                'info': {
                    'original_prompt': user_prompt,
                    'enhanced_prompt': enhanced_prompt,
                    'original_reward': reward,
                    'enhanced_reward': reward
                },
                'is_baseline': is_baseline
            }
            batch_experiences.append(experience)
            logger.info(f"    ✅ 리워드: {reward:.4f}")
        
        return batch_experiences
    
    def generate_batch_images_and_rewards(self, user_prompt: str, enhanced_prompts: List[str]) -> List[float]:
        """배치 이미지 생성 및 리워드 계산 - 모든 프로세스에서 실행 가능"""
        process_info = f"프로세스 {getattr(self, 'process_id', 0)}"
        logger.info(f"🖼️ {process_info}: 배치 이미지 생성 시작 ({len(enhanced_prompts)}개)")
        
        batch_rewards = []
        batch_images = []
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # 프로세스별 GPU 할당 계획
        # 프로세스 0: GPU 5,6 사용 (메인)
        # 프로세스 1: GPU 7,0 사용 
        # 프로세스 2: GPU 1,2 사용
        # 프로세스 3: GPU 3,4 사용
        process_id = getattr(self, 'process_id', 0)
        
        try:
            # 1단계: 배치 이미지 생성 (프로세스별 GPU 사용)
            logger.info(f"🎨 {process_info}: 1단계 - 배치 이미지 생성")
            
            # 각 프로세스가 독립적으로 SD3 파이프라인 생성
            if available_gpus > 0:
                # 프로세스별 GPU 선택
                if process_id == 0:
                    sd_gpu = min(6, available_gpus - 1)  # GPU 6 또는 마지막 GPU
                elif process_id == 1:
                    sd_gpu = min(7, available_gpus - 1) if available_gpus > 7 else 0
                elif process_id == 2:
                    sd_gpu = min(1, available_gpus - 1)
                else:
                    sd_gpu = min(3, available_gpus - 1)
                
                logger.info(f"🎨 {process_info}: GPU {sd_gpu}에서 SD3 사용")
                
                # SD3 파이프라인이 없으면 동적으로 로드
                if not hasattr(self, 'sd_pipeline') or self.sd_pipeline is None:
                    logger.info(f"🔧 {process_info}: SD3 파이프라인 동적 로드 중...")
                    try:
                        from main import load_stable_diffusion_pipeline
                        self.sd_pipeline = load_stable_diffusion_pipeline(device=f"cuda:{sd_gpu}")
                        logger.info(f"✅ {process_info}: SD3 파이프라인 로드 완료")
                    except Exception as e:
                        logger.error(f"❌ {process_info}: SD3 로드 실패: {e}")
                        self.sd_pipeline = None
                
                # 이미지 생성
                if self.sd_pipeline is not None:
                    with torch.cuda.device(sd_gpu):
                        for i, enhanced_prompt in enumerate(enhanced_prompts):
                            enhanced_result = self.sd_pipeline(
                                prompt=enhanced_prompt,
                                num_inference_steps=28,
                                guidance_scale=7.0,
                                height=1024,
                                width=1024
                            )
                            enhanced_image = enhanced_result.images[0]
                            batch_images.append(enhanced_image)
                            logger.info(f"  {process_info}: 이미지 {i+1}/{len(enhanced_prompts)} 생성 완료")
                else:
                    # SD3가 없는 경우 더미 이미지들
                    from PIL import Image
                    for _ in enhanced_prompts:
                        batch_images.append(Image.new('RGB', (1024, 1024), color='black'))
                    logger.warning(f"⚠️ {process_info}: SD3 없음, 더미 이미지 사용")
            else:
                # GPU 없는 경우 더미 이미지
                from PIL import Image
                for _ in enhanced_prompts:
                    batch_images.append(Image.new('RGB', (1024, 1024), color='gray'))
                logger.warning(f"⚠️ {process_info}: GPU 없음, 더미 이미지 사용")
            
            # 2단계: 배치 리워드 계산 (프로세스별 GPU 사용)
            logger.info(f"🎯 {process_info}: 2단계 - 배치 리워드 계산")
            
            if available_gpus > 0:
                # 프로세스별 CLIP GPU 선택
                if process_id == 0:
                    clip_gpu = min(5, available_gpus - 1)  # GPU 5 또는 마지막-1 GPU
                elif process_id == 1:
                    clip_gpu = 0
                elif process_id == 2:
                    clip_gpu = min(2, available_gpus - 1)
                else:
                    clip_gpu = min(4, available_gpus - 1)
                
                logger.info(f"🎯 {process_info}: GPU {clip_gpu}에서 CLIP 사용")
                
                # CLIP 리워드 모델이 없으면 동적으로 로드
                if not hasattr(self, 'reward_model') or self.reward_model is None:
                    logger.info(f"🔧 {process_info}: CLIP 리워드 모델 동적 로드 중...")
                    try:
                        from clip_reward import CLIPReward
                        self.reward_model = CLIPReward(device=f"cuda:{clip_gpu}")
                        logger.info(f"✅ {process_info}: CLIP 모델 로드 완료")
                    except Exception as e:
                        logger.error(f"❌ {process_info}: CLIP 로드 실패: {e}")
                        self.reward_model = None
                
                # 리워드 계산
                if self.reward_model is not None:
                    batch_rewards = self.calculate_batch_clip_rewards(
                        user_prompt, enhanced_prompts, batch_images
                    )
                else:
                    # CLIP가 없는 경우 기본 리워드들
                    batch_rewards = [0.3] * len(enhanced_prompts)
                    logger.warning(f"⚠️ {process_info}: CLIP 없음, 기본 리워드 사용")
            else:
                # GPU 없는 경우 기본 리워드
                batch_rewards = [0.2] * len(enhanced_prompts)
                logger.warning(f"⚠️ {process_info}: GPU 없음, 기본 리워드 사용")
                
        except Exception as e:
            logger.error(f"❌ {process_info}: 배치 이미지 생성 실패: {e}")
            batch_rewards = [0.1] * len(enhanced_prompts)  # 에러 시 낮은 리워드
        
        avg_reward = sum(batch_rewards)/len(batch_rewards) if batch_rewards else 0.0
        logger.info(f"✅ {process_info}: 배치 처리 완료 - 평균 리워드 {avg_reward:.4f}")
        return batch_rewards
    
    def calculate_batch_clip_rewards(self, user_prompt: str, enhanced_prompts: List[str], images: List) -> List[float]:
        """배치 CLIP 리워드 계산 - CLIP 모델의 배치 처리 활용"""
        logger.info(f"🔍 배치 CLIP 리워드 계산 시작 ({len(images)}개)")
        
        batch_rewards = []
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        try:
            if available_gpus > 5 and hasattr(self, 'reward_model') and self.reward_model is not None:
                with torch.cuda.device(5):
                    # CLIP 모델의 배치 리워드 계산 메서드 사용
                    batch_rewards = self.reward_model.calculate_batch_rewards(
                        user_prompt, enhanced_prompts, images
                    )
            else:
                # GPU가 부족하거나 모델이 없는 경우 기본 리워드
                batch_rewards = [0.3] * len(images)
                logger.warning("⚠️ GPU 5번 또는 CLIP 모델 없음, 기본 리워드 사용")
                
        except Exception as e:
            logger.error(f"❌ 배치 CLIP 리워드 계산 실패: {e}")
            batch_rewards = [0.2] * len(images)  # 에러 시 낮은 리워드
        
        logger.info(f"✅ 배치 CLIP 처리 완료: 평균 {sum(batch_rewards)/len(batch_rewards):.4f}")
        return batch_rewards
    
    def compute_grpo_advantages(self, experiences: List[Dict]) -> List[Dict]:
        """GRPO 어드밴티지 계산 (그룹 평균 baseline)"""
        if not experiences:
            return experiences
        
        # 프롬프트별로 그룹화
        prompt_groups = defaultdict(list)
        for exp in experiences:
            prompt_groups[exp['user_prompt']].append(exp)
        
        # 그룹별 어드밴티지 계산
        for prompt, group_exps in prompt_groups.items():
            rewards = [exp['reward'] for exp in group_exps]
            group_baseline = np.mean(rewards)
            
            # 각 경험에 어드밴티지 추가
            for exp in group_exps:
                exp['advantage'] = exp['reward'] - group_baseline
                exp['baseline'] = group_baseline
        
        return experiences
    
    def train_step(self, experiences: List[Dict]) -> Dict:
        """GRPO 학습 스텝"""
        if not experiences:
            return {}
        
        logger.info(f"🎯 GRPO 학습 스텝 시작 (경험: {len(experiences)}개)")
        
        # 어드밴티지 계산
        experiences = self.compute_grpo_advantages(experiences)
        
        # QWEN 모델의 GRPO 업데이트 호출 (Accelerate 지원)
        if self.accelerator:
            # Accelerate 환경에서는 device context 불필요
            metrics = self.qwen_model.update_grpo_policy(experiences)
        else:
            with torch.cuda.device(0):
                metrics = self.qwen_model.update_grpo_policy(experiences)
        
        logger.info(f"✅ GRPO 업데이트 완료")
        logger.info(f"  Policy Loss: {metrics.get('policy_loss', 0):.4f}")
        logger.info(f"  KL Div: {metrics.get('kl_div', 0):.4f}")
        logger.info(f"  Mean Reward: {metrics.get('mean_reward', 0):.4f}")
        
        return metrics
    
    def collect_baseline_data(self, train_prompts: List[str], num_baseline_episodes: int = 3):
        """베이스라인 데이터 수집 (학습에 사용되지 않음)"""
        logger.info(f"📊 베이스라인 데이터 수집 시작 ({num_baseline_episodes} 에피소드)")
        logger.info("=" * 80)
        
        baseline_experiences = []
        
        for episode in range(num_baseline_episodes):
            logger.info(f"\n📋 베이스라인 에피소드 {episode + 1}/{num_baseline_episodes}")
            logger.info("-" * 60)
            
            # 베이스라인용 롤아웃 수집 (저장하지 않음)
            episode_experiences = self.collect_rollouts(train_prompts, is_baseline=True)
            baseline_experiences.extend(episode_experiences)
            
            if episode_experiences:
                episode_rewards = [exp['reward'] for exp in episode_experiences]
                avg_reward = np.mean(episode_rewards)
                logger.info(f"📈 베이스라인 에피소드 {episode + 1} 평균 리워드: {avg_reward:.4f}")
        
        # 베이스라인 통계 계산
        if baseline_experiences:
            baseline_rewards = [exp['reward'] for exp in baseline_experiences]
            baseline_mean = np.mean(baseline_rewards)
            baseline_std = np.std(baseline_rewards)
            
            logger.info("\n📊 베이스라인 통계:")
            logger.info(f"  평균 리워드: {baseline_mean:.4f}")
            logger.info(f"  표준편차: {baseline_std:.4f}")
            logger.info(f"  최대 리워드: {np.max(baseline_rewards):.4f}")
            logger.info(f"  최소 리워드: {np.min(baseline_rewards):.4f}")
            logger.info(f"  총 경험 수: {len(baseline_experiences)}개")
        
        # 환경 에피소드 카운터 리셋 (실제 학습은 1부터 시작)
        self.env.episode_count = 0
        logger.info("\n🔄 환경 에피소드 카운터 리셋 완료")
        logger.info("✅ 베이스라인 데이터 수집 완료!")
        
        return baseline_experiences

    def train(self, train_prompts: List[str], num_epochs: int = 10, num_baseline_episodes: int = 3):
        """GRPO 학습 실행"""
        logger.info(f"🚀 QWEN GRPO 학습 시작")
        logger.info("=" * 80)
        
        # 1단계: 베이스라인 데이터 수집 (학습에 사용되지 않음)
        baseline_data = self.collect_baseline_data(train_prompts, num_baseline_episodes)
        
        logger.info(f"\n🎯 실제 GRPO 학습 시작 (에포크: {num_epochs})")
        logger.info("=" * 80)
        
        all_metrics = []
        
        for epoch in range(num_epochs):
            logger.info(f"\n🔄 학습 에포크 {epoch + 1}/{num_epochs}")
            logger.info("-" * 60)
            
            # 학습용 롤아웃 수집
            experiences = self.collect_rollouts(train_prompts, is_baseline=False)
            
            if not experiences:
                logger.warning("⚠️ 수집된 경험이 없습니다. 다음 에포크로 건너뜁니다.")
                continue
            
            # 학습 스텝
            metrics = self.train_step(experiences)
            metrics['epoch'] = epoch + 1
            all_metrics.append(metrics)
            
            # 에피소드 평균 리워드 계산 및 플롯 업데이트
            epoch_rewards = [exp['reward'] for exp in experiences]
            avg_reward = np.mean(epoch_rewards)
            self._update_reward_plot(epoch + 1, avg_reward)
            
            # 주기적으로 샘플 출력 확인 (다양한 프롬프트 사용)
            if (epoch + 1) % 3 == 0:
                logger.info(f"\n📋 에포크 {epoch + 1} 샘플 출력:")
                # 매번 다른 프롬프트 선택 (순환)
                sample_indices = [(epoch * 2) % len(train_prompts), ((epoch * 2) + 1) % len(train_prompts)]
                sample_prompts = [train_prompts[i] for i in sample_indices]
                self._log_sample_outputs(sample_prompts)
        
        # 최종 플롯 저장
        self._save_reward_plot()
        logger.info("\n✅ QWEN GRPO 학습 완료!")
        return all_metrics, baseline_data
    
    def _log_sample_outputs(self, sample_prompts: List[str]):
        """샘플 출력 로깅"""
        for prompt in sample_prompts:
            try:
                if self.accelerator:
                    # Accelerate 환경에서 샘플 출력
                    basic_result = self.qwen_model.enhance_prompt(prompt)
                    grpo_enhanced, _ = self.qwen_model.generate_grpo_enhanced_prompt(prompt)
                else:
                    with torch.cuda.device(0):
                        # 기본 향상
                        basic_result = self.qwen_model.enhance_prompt(prompt)
                        
                        # GRPO 기반 생성
                        grpo_enhanced, _ = self.qwen_model.generate_grpo_enhanced_prompt(prompt)
                
                logger.info(f"  원본: '{prompt}'")
                logger.info(f"  기본: '{basic_result['enhanced_prompt'][:60]}...'")
                logger.info(f"  GRPO: '{grpo_enhanced[:60]}...'")
                
            except Exception as e:
                logger.warning(f"  샘플 출력 실패: {e}")
    
    def _update_reward_plot(self, epoch: int, avg_reward: float):
        """실시간 리워드 플롯 업데이트"""
        self.episode_numbers.append(epoch)
        self.episode_rewards.append(avg_reward)
        
        # 이동 평균 계산 (윈도우 크기: 5)
        window_size = min(5, len(self.episode_rewards))
        if len(self.episode_rewards) >= window_size:
            running_avg = np.mean(self.episode_rewards[-window_size:])
            self.running_avg_rewards.append(running_avg)
        else:
            self.running_avg_rewards.append(avg_reward)
        
        # 플롯 업데이트
        try:
            plt.clf()  # Clear figure
            
            # 메인 리워드 플롯
            plt.plot(self.episode_numbers, self.episode_rewards, 'b-', alpha=0.6, label='Episode Reward')
            plt.plot(self.episode_numbers, self.running_avg_rewards, 'r-', linewidth=2, label='Moving Average (5)')
            
            plt.title('QWEN GRPO Training Progress', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Average Reward', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Y축 범위 자동 조정
            if len(self.episode_rewards) > 1:
                y_min = min(self.episode_rewards) * 0.95
                y_max = max(self.episode_rewards) * 1.05
                plt.ylim(y_min, y_max)
            
            # 현재 에포크 정보 표시
            plt.text(0.02, 0.98, f'Current Epoch: {epoch}\nCurrent Reward: {avg_reward:.4f}\nMoving Avg: {self.running_avg_rewards[-1]:.4f}', 
                    transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.pause(0.01)  # 짧은 pause로 플롯 업데이트
            
        except Exception as e:
            logger.warning(f"⚠️ 플롯 업데이트 실패: {e}")
    
    def _save_reward_plot(self):
        """최종 리워드 플롯 저장"""
        try:
            if not self.episode_numbers:
                logger.warning("⚠️ 저장할 리워드 데이터가 없습니다.")
                return
            
            # 최종 플롯 생성
            plt.figure(figsize=(12, 8))
            
            # 서브플롯 1: 리워드 추이
            plt.subplot(2, 1, 1)
            plt.plot(self.episode_numbers, self.episode_rewards, 'b-', alpha=0.6, marker='o', markersize=4, label='Episode Reward')
            plt.plot(self.episode_numbers, self.running_avg_rewards, 'r-', linewidth=3, label='Moving Average (5)')
            plt.title('QWEN GRPO Training Progress - Reward Trend', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Average Reward')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 서브플롯 2: 리워드 분포 히스토그램
            plt.subplot(2, 1, 2)
            plt.hist(self.episode_rewards, bins=min(20, len(self.episode_rewards)), alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Reward Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Reward Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # 통계 정보 추가
            mean_reward = np.mean(self.episode_rewards)
            std_reward = np.std(self.episode_rewards)
            max_reward = np.max(self.episode_rewards)
            min_reward = np.min(self.episode_rewards)
            
            stats_text = f'Statistics:\nMean: {mean_reward:.4f}\nStd: {std_reward:.4f}\nMax: {max_reward:.4f}\nMin: {min_reward:.4f}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            plt.tight_layout()
            
            # 저장
            if self.config.save_images:
                plot_path = os.path.join(self.plot_save_dir, 'training_progress.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"📊 학습 진행 플롯 저장: {plot_path}")
                
                # 데이터도 CSV로 저장
                import csv
                csv_path = os.path.join(self.plot_save_dir, 'training_rewards.csv')
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Epoch', 'Average_Reward', 'Moving_Average'])
                    for i in range(len(self.episode_numbers)):
                        writer.writerow([self.episode_numbers[i], self.episode_rewards[i], self.running_avg_rewards[i]])
                logger.info(f"📊 학습 데이터 저장: {csv_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"❌ 최종 플롯 저장 실패: {e}")

# ... existing code ...

 