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
    """QWEN GRPO 통합 환경"""
    
    def __init__(self, qwen_model: QWENModel, reward_model, sd_pipeline, config: QWENGRPOConfig):
        self.qwen_model = qwen_model
        self.reward_model = reward_model
        self.sd_pipeline = sd_pipeline
        self.config = config
        
        # GPU 디바이스 설정
        self.qwen_device = "cuda:0"    # QWEN (프롬프트 향상)
        self.sd_device = "cuda:1"      # Stable Diffusion (이미지 생성)
        self.reward_device = "cuda:2"  # CLIP Reward (리워드 계산)
        
        self.current_user_prompt = ""
        self.current_enhanced_prompt = ""
        self.current_candidates = []
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
        self.current_candidates = []
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
            'candidates': [],
            'episode': self.episode_count
        }
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """환경 스텝 - QWEN 후보 중에서 선택"""
        original_image = None
        enhanced_image = None
        original_reward = 0.0
        enhanced_reward = 0.0
        total_reward = 0.0
        candidates = []
        
        try:
            # QWEN에서 향상된 프롬프트 후보들 생성
            logger.info(f"🧠 QWEN 후보 생성 중... (GPU {self.qwen_device})")
            with torch.cuda.device(0):
                candidates = self.qwen_model.generate_enhancement_candidates(self.current_user_prompt)
            
            self.current_candidates = candidates
            
            # 액션에 해당하는 후보 선택
            if 0 <= action < len(candidates):
                selected_prompt = candidates[action]
            else:
                logger.warning(f"Invalid action {action}, using first candidate")
                selected_prompt = candidates[0] if candidates else self.current_user_prompt
            
            self.current_enhanced_prompt = selected_prompt
            
            logger.info(f"✅ 선택된 프롬프트: '{selected_prompt[:50]}...'")
            
            # 이미지 생성 시도
            try:
                logger.info(f"🖼️  이미지 생성 시작 (GPU {self.sd_device})")
                
                with torch.cuda.device(1):
                    # 원본 프롬프트로 이미지 생성
                    original_result = self.sd_pipeline(
                        prompt=self.current_user_prompt,
                        num_inference_steps=28,
                        guidance_scale=7.0,
                        height=1024,
                        width=1024
                    )
                    original_image = original_result.images[0]
                    
                    # 향상된 프롬프트로 이미지 생성
                    enhanced_result = self.sd_pipeline(
                        prompt=self.current_enhanced_prompt,
                        num_inference_steps=28,
                        guidance_scale=7.0,
                        height=1024,
                        width=1024
                    )
                    enhanced_image = enhanced_result.images[0]
                
                logger.info(f"✅ 이미지 생성 완료")
                
            except Exception as img_error:
                logger.error(f"❌ 이미지 생성 실패: {img_error}")
                # 더미 이미지 생성 (검은 이미지)
                from PIL import Image
                original_image = Image.new('RGB', (1024, 1024), color='black')
                enhanced_image = Image.new('RGB', (1024, 1024), color='black')
            
            # 리워드 계산 시도
            try:
                logger.info(f"🎯 리워드 계산 시작 (GPU {self.reward_device})")
                
                # CLIP 리워드를 GPU 2에서 계산
                with torch.cuda.device(2):
                    enhanced_reward = self.reward_model.calculate_reward(
                        self.current_user_prompt,
                        self.current_enhanced_prompt,
                        enhanced_image
                    )
                    
                    # 원본 프롬프트 vs 원본 이미지 (참고용)
                    original_reward = self.reward_model.calculate_reward(
                        self.current_user_prompt,
                        self.current_user_prompt,
                        original_image
                    )
                
                # 리워드 계산
                total_reward = enhanced_reward
                
                logger.info(f"✅ 리워드 계산 완료: {total_reward:.4f}")
                
            except Exception as reward_error:
                logger.error(f"❌ 리워드 계산 실패: {reward_error}")
                # 기본 리워드 값 사용
                original_reward = 0.1
                enhanced_reward = 0.1
                total_reward = 0.1
            
            # 다음 상태 (에피소드 완료)
            next_state = {
                'user_prompt': self.current_user_prompt,
                'enhanced_prompt': self.current_enhanced_prompt,
                'candidates': self.current_candidates,
                'episode': self.episode_count
            }
            
            info = {
                'original_prompt': self.current_user_prompt,
                'enhanced_prompt': self.current_enhanced_prompt,
                'candidates': self.current_candidates,
                'selected_action': action,
                'original_reward': original_reward,
                'enhanced_reward': enhanced_reward
            }
            
            # 한 스텝으로 완료 (done=True)
            return next_state, total_reward, True, info
            
        except Exception as e:
            logger.error(f"❌ 전체 스텝 실패: {e}")
            import traceback
            traceback.print_exc()
            
            # 에러가 발생해도 기본값으로 상태 반환
            if not candidates:
                candidates = [self.current_user_prompt]
            
            next_state = {
                'user_prompt': self.current_user_prompt,
                'enhanced_prompt': self.current_user_prompt,
                'candidates': candidates,
                'episode': self.episode_count
            }
            
            info = {
                'original_prompt': self.current_user_prompt,
                'enhanced_prompt': self.current_user_prompt,
                'candidates': candidates,
                'selected_action': 0,
                'original_reward': 0.0,
                'enhanced_reward': 0.0,
                'error': str(e)
            }
            
            return next_state, 0.0, True, info
        
        finally:
            # 에러 발생 여부와 관계없이 항상 이미지 저장 시도
            if self.config.save_images and original_image is not None and enhanced_image is not None:
                try:
                    self._save_episode_results(
                        original_image, enhanced_image, 
                        original_reward, enhanced_reward, total_reward,
                        action, candidates
                    )
                except Exception as save_error:
                    logger.warning(f"⚠️ 이미지 저장 실패: {save_error}")
                    # 에러 정보라도 저장
                    self._save_error_log(action, candidates, str(save_error))
    
    def _save_episode_results(self, original_image, enhanced_image, original_reward, enhanced_reward, total_reward, action, candidates):
        """에피소드 결과 저장"""
        try:
            # 이미지 저장
            original_image.save(os.path.join(self.episode_dir, "original_image.png"))
            enhanced_image.save(os.path.join(self.episode_dir, "enhanced_image.png"))
            
            # 후보들 정보
            candidates_info = "\n".join([f"  {i}: {cand}" for i, cand in enumerate(candidates)])
            
            # 로그 파일 작성
            log_content = f"""=== QWEN GRPO Episode Results ===
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Episode: {self.episode_count}

=== Prompts ===
Original Prompt: {self.current_user_prompt}
Enhanced Prompt: {self.current_enhanced_prompt}

=== GRPO Action ===
Selected Action: {action}
Available Candidates ({len(candidates)}):
{candidates_info}

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
    
    def _save_error_log(self, action: int, candidates: List[str], error_msg: str):
        """에러 발생 시 로그만 저장"""
        try:
            # 에러 로그 파일 작성
            error_log_content = f"""=== QWEN GRPO Error Log ===
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Episode: {self.episode_count}

=== Error Info ===
Error Message: {error_msg}
Original Prompt: {self.current_user_prompt}
Selected Action: {action}

=== Available Candidates ===
{chr(10).join([f"  {i}: {cand}" for i, cand in enumerate(candidates)])}

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
    """QWEN 통합 GRPO 트레이너"""
    
    def __init__(self, qwen_model: QWENModel, reward_model, sd_pipeline, config: QWENGRPOConfig):
        self.config = config
        self.qwen_model = qwen_model
        self.reward_model = reward_model
        self.sd_pipeline = sd_pipeline
        
        # QWEN 모델에 GRPO 컴포넌트가 설정되어 있는지 확인
        if not hasattr(qwen_model, 'grpo_policy_head'):
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
        logger.info(f"✅ Action Space: {config.num_enhancement_candidates} enhancement candidates")
        logger.info(f"📊 플롯 저장 디렉토리: {self.plot_save_dir}")
    
    def collect_rollouts(self, prompts: List[str], is_baseline: bool = False) -> List[Dict]:
        """롤아웃 수집"""
        all_experiences = []
        rollout_type = "베이스라인" if is_baseline else "학습용"
        
        for prompt_idx, user_prompt in enumerate(prompts):
            logger.info(f"\n📝 {rollout_type} 프롬프트 {prompt_idx + 1}/{len(prompts)}: '{user_prompt}'")
            
            # 프롬프트별 롤아웃 수집
            for rollout_idx in range(self.config.num_rollouts):
                logger.info(f"  🎲 {rollout_type} 롤아웃 {rollout_idx + 1}/{self.config.num_rollouts}")
                
                try:
                    # 환경 리셋 (베이스라인일 때는 이미지 저장 안함)
                    if is_baseline:
                        # 베이스라인 수집 시에는 이미지 저장 비활성화
                        original_save_setting = self.config.save_images
                        self.config.save_images = False
                    
                    state = self.env.reset(user_prompt)
                    
                    # QWEN GRPO로 액션 선택
                    with torch.cuda.device(0):
                        action, log_prob, candidates = self.qwen_model.get_grpo_action_and_log_prob(user_prompt)
                    
                    logger.info(f"    🎯 선택된 액션: {action} (로그 확률: {log_prob:.4f})")
                    
                    # 환경 스텝 실행
                    next_state, reward, done, info = self.env.step(action)
                    
                    # 베이스라인일 때는 설정 복원
                    if is_baseline:
                        self.config.save_images = original_save_setting
                    
                    if next_state is not None:
                        # 경험 저장
                        experience = {
                            'user_prompt': user_prompt,
                            'action': action,
                            'log_prob': log_prob,
                            'reward': reward,
                            'candidates': candidates,
                            'info': info,
                            'is_baseline': is_baseline
                        }
                        
                        all_experiences.append(experience)
                        logger.info(f"    ✅ {rollout_type} 리워드: {reward:.4f}")
                    else:
                        logger.warning(f"    ❌ {rollout_type} 롤아웃 실패")
                
                except Exception as e:
                    logger.error(f"    ❌ {rollout_type} 롤아웃 오류: {e}")
                    # 베이스라인일 때 설정 복원 (에러 상황에서도)
                    if is_baseline and 'original_save_setting' in locals():
                        self.config.save_images = original_save_setting
                    continue
        
        logger.info(f"\n📊 수집된 {rollout_type} 경험: {len(all_experiences)}개")
        return all_experiences
    
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
        
        # QWEN 모델의 GRPO 업데이트 호출
        with torch.cuda.device(0):
            metrics = self.qwen_model.update_grpo_policy(experiences)
        
        logger.info(f"✅ GRPO 업데이트 완료")
        logger.info(f"  Policy Loss: {metrics.get('policy_loss', 0):.4f}")
        logger.info(f"  KL Div: {metrics.get('kl_div', 0):.4f}")
        logger.info(f"  Entropy: {metrics.get('entropy', 0):.4f}")
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
                with torch.cuda.device(0):
                    # 기본 향상
                    basic_result = self.qwen_model.enhance_prompt(prompt)
                    
                    # GRPO 기반 선택
                    action, log_prob, candidates = self.qwen_model.get_grpo_action_and_log_prob(prompt)
                    grpo_enhanced = candidates[action] if 0 <= action < len(candidates) else candidates[0]
                
                logger.info(f"  원본: '{prompt}'")
                logger.info(f"  기본: '{basic_result['enhanced_prompt'][:60]}...'")
                logger.info(f"  GRPO: '{grpo_enhanced[:60]}...' (액션: {action})")
                
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

 