#!/usr/bin/env python3
"""
Simple Multi-GPU GRPO Training
GPU 0: QWEN LoRA Training
GPU 1: CLIP Reward Calculation  
GPU 2: Stable Diffusion 3 Image Generation
"""

import torch
import logging
import os
from typing import List, Dict
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from qwen import QWENModel, QWENGRPOConfig
from clip_reward import CLIPReward

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('grpo_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_stable_diffusion_pipeline(device="cuda:2"):
    """Stable Diffusion 3 파이프라인 로드 (GPU 2번 전용)"""
    try:
        from diffusers import StableDiffusion3Pipeline
        
        logger.info(f"🎨 SD3 파이프라인 로딩 중... ({device})")
        
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        
        # 지정된 GPU로 이동
        pipe = pipe.to(device)
        logger.info(f"✅ SD3 파이프라인을 {device}로 이동")
        
        # 메모리 최적화
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.set_progress_bar_config(disable=True)
        
        logger.info("✅ Stable Diffusion 3 파이프라인 로드 완료")
        return pipe
        
    except Exception as e:
        logger.error(f"❌ SD3 파이프라인 로드 실패: {e}")
        raise

def get_training_prompts():
    """학습용 프롬프트 데이터셋"""
    import random
    
    selected_prompts = [
        "a beautiful cat sitting on a chair",
        "sunset over mountains with golden light", 
        "abstract art painting with vibrant colors",
        "portrait of a woman with flowing hair",
        "futuristic city skyline at night",
        "red apple on blue table with green background",
        "transparent glass sphere floating in purple space",
        "crowded marketplace with many people and colorful stalls"
    ]
    
    random.shuffle(selected_prompts)
    return selected_prompts

class SimpleGRPOTrainer:
    """간단한 멀티 GPU GRPO 트레이너"""
    
    def __init__(self, config: QWENGRPOConfig):
        self.config = config
        
        # 로깅 디렉토리 설정
        self.log_dir = config.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "plots"), exist_ok=True)
        
        # 학습 메트릭 추적
        self.training_metrics = {
            'epoch_rewards': [],
            'policy_losses': [],
            'kl_divergences': [],
            'advantages': [],
            'epoch_times': []
        }
        
        # GPU 가용성 확인
        if not torch.cuda.is_available():
            logger.warning("⚠️ CUDA 사용 불가 - CPU로 실행")
            self.use_gpu = False
        elif torch.cuda.device_count() < 3:
            logger.warning(f"⚠️ GPU {torch.cuda.device_count()}개만 사용 가능 (권장: 3개)")
            self.use_gpu = True
        else:
            self.use_gpu = True
            logger.info(f"🎯 사용 가능한 GPU: {torch.cuda.device_count()}개")
            for i in range(min(3, torch.cuda.device_count())):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # 모델들 초기화
        self._init_models()
        
    def _init_models(self):
        """각 GPU별 모델 초기화 - 메모리 최적화"""
        logger.info("🔧 모델들 초기화 중... (메모리 최적화)")
        
        # GPU 0: QWEN LoRA 모델만 - 더 작은 모델 사용
        qwen_device = "cuda:0" if self.use_gpu else "cpu"
        logger.info(f"🧠 {qwen_device}: QWEN LoRA 모델 로딩... (더 작은 모델)")
        self.qwen_model = QWENModel(
            model_name="Qwen/Qwen2-VL-2B-Instruct",  # 7B → 2B로 변경
            device=qwen_device,
            temperature=0.7,
            grpo_config=self.config
        )
        logger.info(f"✅ QWEN 2B 모델 로드 완료 ({qwen_device})")
        
        # GPU 1: CLIP 리워드 모델만
        clip_device = "cuda:1" if self.use_gpu and torch.cuda.device_count() > 1 else "cuda:0" if self.use_gpu else "cpu"
        logger.info(f"🎯 {clip_device}: CLIP 리워드 모델 로딩...")
        self.reward_model = CLIPReward(device=clip_device)
        self.clip_device = clip_device
        logger.info(f"✅ CLIP 모델 로드 완료 ({clip_device})")
        
        # GPU 2: Stable Diffusion 3만 (이미지 생성 전용)
        sd_device = "cuda:2" if self.use_gpu and torch.cuda.device_count() > 2 else "cuda:0" if self.use_gpu else "cpu"
        logger.info(f"🎨 {sd_device}: SD3 파이프라인 로딩... (이미지 생성 전용)")
        self.sd_pipeline = load_stable_diffusion_pipeline(device=sd_device)
        self.sd_device = sd_device
        logger.info(f"✅ SD3 파이프라인 로드 완료 ({sd_device})")
        
        # GPU 메모리 정리
        if self.use_gpu:
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
            logger.info("🧹 초기화 후 GPU 메모리 정리 완료")
        
        logger.info("🎯 모든 모델 초기화 완료!")
        logger.info("📋 GPU 할당:")
        logger.info(f"  GPU 0: QWEN 2B LoRA 모델 ({qwen_device})")
        logger.info(f"  GPU 1: CLIP 리워드 모델 + Reference 모델 ({clip_device})")
        logger.info(f"  GPU 2: SD3 이미지 생성 ({sd_device})")
        logger.info("🔄 이미지는 GPU 2에서 생성 후 GPU 1로 이동하여 리워드 계산")
        logger.info("🔄 Reference 모델은 GPU 1에서 KL penalty 계산")
        
    def generate_enhanced_prompts(self, user_prompt: str, num_rollouts: int) -> List[tuple]:
        """향상된 프롬프트 생성 (GPU 0에서 실행)"""
        logger.info(f"🧠 향상된 프롬프트 생성 ({num_rollouts}개)")
        
        enhanced_data = []
        
        for i in range(num_rollouts):
            try:
                enhanced_prompt, log_prob = self.qwen_model.generate_grpo_enhanced_prompt(user_prompt)
                enhanced_data.append((enhanced_prompt, log_prob))
                logger.info(f"  생성 {i+1}/{num_rollouts}: '{enhanced_prompt[:50]}...' (log_prob: {log_prob:.4f})")
            except Exception as e:
                logger.error(f"  생성 {i+1} 실패: {e}")
                continue
        
        logger.info(f"✅ {len(enhanced_data)}개 프롬프트 생성 완료")
        return enhanced_data
    
    def generate_images(self, enhanced_prompts: List[str]) -> List:
        """이미지 생성 (GPU 2에서 실행) - 단일 처리"""
        logger.info(f"🎨 이미지 생성 ({len(enhanced_prompts)}개)")
        
        images = []
        
        for i, prompt in enumerate(enhanced_prompts):
            try:
                result = self.sd_pipeline(
                    prompt=prompt,
                    num_inference_steps=28,
                    guidance_scale=7.0,
                    height=1024,
                    width=1024
                )
                image = result.images[0]
                images.append(image)
                logger.info(f"  이미지 {i+1}/{len(enhanced_prompts)} 생성 완료")
            except Exception as e:
                logger.error(f"  이미지 {i+1} 생성 실패: {e}")
                # 더미 이미지 추가
                from PIL import Image
                images.append(Image.new('RGB', (1024, 1024), color='black'))
        
        logger.info(f"✅ {len(images)}개 이미지 생성 완료")
        return images
    
    def generate_images_batch(self, enhanced_prompts: List[str]) -> List:
        """배치 이미지 생성 (GPU 2에서 실행) - 배치 최적화"""
        logger.info(f"🎨 배치 이미지 생성 ({len(enhanced_prompts)}개)")
        
        images = []
        
        try:
            # SD3는 배치 처리를 지원하지만 메모리 절약을 위해 개별 처리
            for i, prompt in enumerate(enhanced_prompts):
                try:
                    result = self.sd_pipeline(
                        prompt=prompt,
                        num_inference_steps=28,
                        guidance_scale=7.0,
                        height=1024,
                        width=1024
                    )
                    image = result.images[0]
                    images.append(image)
                    logger.info(f"  배치 이미지 {i+1}/{len(enhanced_prompts)} 생성 완료")
                except Exception as e:
                    logger.error(f"  배치 이미지 {i+1} 생성 실패: {e}")
                    # 더미 이미지 추가
                    from PIL import Image
                    images.append(Image.new('RGB', (1024, 1024), color='black'))
        
        except Exception as e:
            logger.error(f"❌ 배치 이미지 생성 전체 실패: {e}")
            # 전체 실패시 더미 이미지들
            from PIL import Image
            images = [Image.new('RGB', (1024, 1024), color='black') for _ in enhanced_prompts]
        
        logger.info(f"✅ 배치 {len(images)}개 이미지 생성 완료")
        return images
    
    def calculate_rewards(self, user_prompt: str, enhanced_prompts: List[str], images: List) -> List[float]:
        """리워드 계산 (GPU 1에서 실행) - 단일 처리"""
        logger.info(f"🎯 리워드 계산 ({len(images)}개)")
        
        rewards = []
        
        for i, (enhanced_prompt, image) in enumerate(zip(enhanced_prompts, images)):
            try:
                reward = self.reward_model.calculate_reward(
                    user_prompt, enhanced_prompt, image
                )
                rewards.append(reward)
                logger.info(f"  리워드 {i+1}/{len(images)}: {reward:.4f}")
            except Exception as e:
                logger.error(f"  리워드 {i+1} 계산 실패: {e}")
                rewards.append(0.1)  # 기본 리워드
        
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        logger.info(f"✅ 리워드 계산 완료 - 평균: {avg_reward:.4f}")
        return rewards
    
    def calculate_rewards_batch(self, user_prompt: str, enhanced_prompts: List[str], images: List) -> List[float]:
        """배치 리워드 계산 - 이미지를 CLIP GPU로 이동"""
        logger.info(f"🎯 배치 리워드 계산 ({len(images)}개) - Original User Prompt 사용")
        logger.info(f"📝 사용된 Original Prompt: '{user_prompt}'")
        logger.info(f"🔄 이미지를 GPU {self.sd_device} → {self.clip_device}로 이동")
        
        try:
            # 이미지들은 이미 PIL Image 형태이므로 직접 CLIP으로 전달 가능
            # (PIL Image는 CPU 메모리에 있으므로 GPU 간 이동 불필요)
            
            # CLIP 배치 처리 사용 - original user prompt로 계산
            rewards = self.reward_model.calculate_batch_rewards(
                user_prompt,  # ⭐ 중요: original user prompt 사용 (enhanced 아님)
                enhanced_prompts,
                images  # PIL Images - CLIP에서 자동으로 적절한 GPU로 처리
            )
            
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            logger.info(f"✅ 배치 리워드 계산 완료 - 평균: {avg_reward:.4f}")
            logger.info(f"🔍 CLIP 유사도는 Original User Prompt '{user_prompt}'와 생성된 이미지 간 계산됨")
            logger.info(f"📊 이미지 처리: SD3 GPU {self.sd_device} → CLIP GPU {self.clip_device}")
            
            return rewards
            
        except Exception as e:
            logger.error(f"❌ 배치 리워드 계산 실패: {e}")
            # 에러 시 개별 계산으로 fallback
            logger.info("🔄 개별 리워드 계산으로 fallback")
            return self.calculate_rewards(user_prompt, enhanced_prompts, images)
    
    def collect_rollouts(self, prompts: List[str]) -> List[Dict]:
        """배치 롤아웃 수집 (Group-relative 방식)"""
        all_experiences = []
        
        # 배치 단위로 처리
        batch_size = min(len(prompts), self.config.batch_size)
        
        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start:batch_start + batch_size]
            logger.info(f"\n📦 배치 {batch_start//batch_size + 1}: {len(batch_prompts)}개 프롬프트")
            
            batch_experiences = self.collect_batch_rollouts(batch_prompts)
            all_experiences.extend(batch_experiences)
        
        logger.info(f"\n📊 총 수집된 경험: {len(all_experiences)}개")
        return all_experiences
    
    def collect_batch_rollouts(self, batch_prompts: List[str]) -> List[Dict]:
        """단일 배치 롤아웃 수집"""
        batch_experiences = []
        
        for prompt_idx, user_prompt in enumerate(batch_prompts):
            logger.info(f"📝 프롬프트 {prompt_idx + 1}/{len(batch_prompts)}: '{user_prompt}'")
            
            # 1단계: 향상된 프롬프트 생성 (GPU 0)
            enhanced_data = self.generate_enhanced_prompts(user_prompt, self.config.num_rollouts)
            
            if not enhanced_data:
                logger.warning(f"⚠️ 프롬프트 생성 실패, 건너뛰기")
                continue
            
            enhanced_prompts = [data[0] for data in enhanced_data]
            log_probs = [data[1] for data in enhanced_data]
            
            # 2단계: 배치 이미지 생성 (GPU 2)  
            images = self.generate_images_batch(enhanced_prompts)
            
            # 3단계: 배치 리워드 계산 (GPU 1) - original user prompt 사용
            rewards = self.calculate_rewards_batch(user_prompt, enhanced_prompts, images)
            
            # 경험 저장
            for enhanced_prompt, log_prob, reward in zip(enhanced_prompts, log_probs, rewards):
                experience = {
                    'user_prompt': user_prompt,
                    'enhanced_prompt': enhanced_prompt,
                    'log_prob': log_prob,
                    'reward': reward,
                    'info': {
                        'original_prompt': user_prompt,
                        'enhanced_prompt': enhanced_prompt,
                        'clip_reward': reward  # CLIP은 original user prompt로 계산됨
                    }
                }
                batch_experiences.append(experience)
        
        return batch_experiences
    
    def collect_rollouts_with_logging(self, prompts: List[str], epoch: int) -> List[Dict]:
        """로깅 기능이 포함된 배치 롤아웃 수집"""
        all_experiences = []
        
        # 배치 단위로 처리
        batch_size = min(len(prompts), self.config.batch_size)
        
        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start:batch_start + batch_size]
            logger.info(f"\n📦 에포크 {epoch} 배치 {batch_start//batch_size + 1}: {len(batch_prompts)}개 프롬프트")
            
            batch_experiences = []
            
            for prompt_idx, user_prompt in enumerate(batch_prompts):
                logger.info(f"📝 프롬프트 {prompt_idx + 1}/{len(batch_prompts)}: '{user_prompt}'")
                
                # 1단계: 향상된 프롬프트 생성 (GPU 0)
                enhanced_data = self.generate_enhanced_prompts(user_prompt, self.config.num_rollouts)
                
                if not enhanced_data:
                    logger.warning(f"⚠️ 프롬프트 생성 실패, 건너뛰기")
                    continue
                
                enhanced_prompts = [data[0] for data in enhanced_data]
                log_probs = [data[1] for data in enhanced_data]
                
                # 2단계: 배치 이미지 생성 (GPU 2)  
                images = self.generate_images_batch(enhanced_prompts)
                
                # 3단계: 배치 리워드 계산 (GPU 1) - original user prompt 사용
                rewards = self.calculate_rewards_batch(user_prompt, enhanced_prompts, images)
                
                # 4단계: 이미지 저장 (로깅)
                if self.config.save_images:
                    self.save_episode_images(epoch, user_prompt, enhanced_prompts, images, rewards)
                
                # 경험 저장
                for enhanced_prompt, log_prob, reward in zip(enhanced_prompts, log_probs, rewards):
                    experience = {
                        'user_prompt': user_prompt,
                        'enhanced_prompt': enhanced_prompt,
                        'log_prob': log_prob,
                        'reward': reward,
                        'info': {
                            'original_prompt': user_prompt,
                            'enhanced_prompt': enhanced_prompt,
                            'clip_reward': reward,  # CLIP은 original user prompt로 계산됨
                            'epoch': epoch
                        }
                    }
                    batch_experiences.append(experience)
            
            all_experiences.extend(batch_experiences)
        
        logger.info(f"\n📊 에포크 {epoch} 총 수집된 경험: {len(all_experiences)}개")
        return all_experiences
    
    def update_policy(self, experiences: List[Dict]) -> Dict:
        """Group-relative 정책 업데이트 (GPU 0에서 실행)"""
        logger.info(f"🔄 Group-relative 정책 업데이트 ({len(experiences)}개 경험)")
        
        # Group-relative baseline 계산
        group_baseline_metrics = self.calculate_group_relative_baseline(experiences)
        
        # 향상된 경험 데이터로 정책 업데이트
        enhanced_experiences = self.apply_group_relative_advantages(experiences, group_baseline_metrics)
        
        # QWEN 모델 업데이트
        metrics = self.qwen_model.update_grpo_policy(enhanced_experiences)
        
        # Group-relative 메트릭 추가
        metrics.update(group_baseline_metrics)
        
        logger.info(f"✅ Group-relative 정책 업데이트 완료")
        return metrics
    
    def calculate_group_relative_baseline(self, experiences: List[Dict]) -> Dict:
        """Group-relative baseline 계산"""
        if not experiences:
            return {}
        
        # 그룹별 리워드 수집
        user_prompt_groups = {}
        for exp in experiences:
            user_prompt = exp['user_prompt']
            if user_prompt not in user_prompt_groups:
                user_prompt_groups[user_prompt] = []
            user_prompt_groups[user_prompt].append(exp['reward'])
        
        # 그룹별 통계 계산
        group_stats = {}
        all_rewards = []
        
        for user_prompt, rewards in user_prompt_groups.items():
            group_mean = sum(rewards) / len(rewards)
            group_std = (sum((r - group_mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
            
            group_stats[user_prompt] = {
                'mean': group_mean,
                'std': group_std,
                'count': len(rewards),
                'rewards': rewards
            }
            all_rewards.extend(rewards)
        
        # 전체 통계
        global_mean = sum(all_rewards) / len(all_rewards)
        global_std = (sum((r - global_mean) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5
        
        logger.info(f"📊 Group-relative Baseline 통계:")
        logger.info(f"  전체 평균: {global_mean:.4f} ± {global_std:.4f}")
        logger.info(f"  그룹 수: {len(group_stats)}")
        
        for prompt, stats in group_stats.items():
            logger.info(f"  '{prompt[:30]}...': {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
        
        return {
            'global_mean': global_mean,
            'global_std': global_std,
            'group_stats': group_stats,
            'num_groups': len(group_stats)
        }
    
    def apply_group_relative_advantages(self, experiences: List[Dict], baseline_metrics: Dict) -> List[Dict]:
        """Group-relative advantage 적용"""
        if not baseline_metrics:
            return experiences
        
        enhanced_experiences = []
        group_stats = baseline_metrics['group_stats']
        global_mean = baseline_metrics['global_mean']
        
        for exp in experiences:
            user_prompt = exp['user_prompt']
            reward = exp['reward']
            
            # Group-relative advantage 계산
            if user_prompt in group_stats:
                group_mean = group_stats[user_prompt]['mean']
                group_advantage = reward - group_mean
            else:
                group_advantage = reward - global_mean
            
            # Global advantage도 계산
            global_advantage = reward - global_mean
            
            # 경험 데이터에 advantage 정보 추가
            enhanced_exp = exp.copy()
            enhanced_exp['group_advantage'] = group_advantage
            enhanced_exp['global_advantage'] = global_advantage
            enhanced_exp['group_baseline'] = group_stats.get(user_prompt, {}).get('mean', global_mean)
            enhanced_exp['global_baseline'] = global_mean
            
            enhanced_experiences.append(enhanced_exp)
        
        logger.info(f"✅ Group-relative advantage 적용 완료")
        return enhanced_experiences
    
    def save_episode_images(self, epoch: int, user_prompt: str, enhanced_prompts: List[str], 
                           images: List, rewards: List[float]):
        """에피소드별 이미지 저장"""
        try:
            epoch_dir = os.path.join(self.log_dir, "images", f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            
            for i, (enhanced_prompt, image, reward) in enumerate(zip(enhanced_prompts, images, rewards)):
                # 이미지 저장
                image_path = os.path.join(epoch_dir, f"image_{i}_reward_{reward:.3f}.png")
                image.save(image_path)
                
                # 프롬프트 정보 저장
                info_path = os.path.join(epoch_dir, f"image_{i}_info.txt")
                with open(info_path, 'w', encoding='utf-8') as f:
                    f.write(f"Original Prompt: {user_prompt}\n")
                    f.write(f"Enhanced Prompt: {enhanced_prompt}\n")
                    f.write(f"CLIP Reward: {reward:.4f}\n")
                    f.write(f"Epoch: {epoch}\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            logger.info(f"💾 에피소드 {epoch} 이미지 저장 완료: {len(images)}개")
            
        except Exception as e:
            logger.error(f"❌ 이미지 저장 실패: {e}")
    
    def plot_training_metrics(self, epoch: int):
        """학습 메트릭 플롯 생성"""
        try:
            if not self.training_metrics['epoch_rewards']:
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'GRPO Training Metrics - Epoch {epoch}', fontsize=16)
            
            # 1. 리워드 추이
            axes[0, 0].plot(self.training_metrics['epoch_rewards'], 'b-o', linewidth=2, markersize=6)
            axes[0, 0].set_title('Average Reward per Epoch')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Average Reward')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Policy Loss 추이
            if self.training_metrics['policy_losses']:
                axes[0, 1].plot(self.training_metrics['policy_losses'], 'r-o', linewidth=2, markersize=6)
                axes[0, 1].set_title('Policy Loss per Epoch')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Policy Loss')
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. KL Divergence 추이
            if self.training_metrics['kl_divergences']:
                axes[1, 0].plot(self.training_metrics['kl_divergences'], 'g-o', linewidth=2, markersize=6)
                axes[1, 0].set_title('KL Divergence per Epoch')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('KL Divergence')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Advantage 분포 (최근 에피소드)
            if self.training_metrics['advantages']:
                recent_advantages = self.training_metrics['advantages'][-50:]  # 최근 50개
                axes[1, 1].hist(recent_advantages, bins=20, alpha=0.7, color='purple', edgecolor='black')
                axes[1, 1].set_title('Recent Advantage Distribution')
                axes[1, 1].set_xlabel('Advantage Value')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 플롯 저장
            plot_path = os.path.join(self.log_dir, "plots", f"training_metrics_epoch_{epoch}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 에피소드 {epoch} 메트릭 플롯 저장 완료")
            
        except Exception as e:
            logger.error(f"❌ 플롯 생성 실패: {e}")
    
    def log_epoch_metrics(self, epoch: int, experiences: List[Dict], metrics: Dict):
        """에피소드 메트릭 로깅"""
        # 리워드 통계
        rewards = [exp['reward'] for exp in experiences]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        
        # 메트릭 저장
        self.training_metrics['epoch_rewards'].append(avg_reward)
        if 'policy_loss' in metrics:
            self.training_metrics['policy_losses'].append(metrics['policy_loss'])
        if 'kl_div' in metrics:
            self.training_metrics['kl_divergences'].append(metrics['kl_div'])
        
        # Advantage 저장
        for exp in experiences:
            if 'group_advantage' in exp:
                self.training_metrics['advantages'].append(exp['group_advantage'])
        
        # 상세 로그 출력
        logger.info(f"📈 에피소드 {epoch} 상세 메트릭:")
        logger.info(f"  📊 평균 리워드: {avg_reward:.4f}")
        logger.info(f"  📊 리워드 범위: {min(rewards):.4f} ~ {max(rewards):.4f}")
        logger.info(f"  📊 경험 수: {len(experiences)}")
        
        if metrics:
            for key, value in metrics.items():
                logger.info(f"  📊 {key}: {value:.4f}")
        
        # CSV 로그 저장
        csv_path = os.path.join(self.log_dir, "training_log.csv")
        if not os.path.exists(csv_path):
            with open(csv_path, 'w') as f:
                f.write("epoch,avg_reward,min_reward,max_reward,num_experiences,policy_loss,kl_div\n")
        
        with open(csv_path, 'a') as f:
            policy_loss = metrics.get('policy_loss', 0.0)
            kl_div = metrics.get('kl_div', 0.0)
            f.write(f"{epoch},{avg_reward:.4f},{min(rewards):.4f},{max(rewards):.4f},"
                   f"{len(experiences)},{policy_loss:.4f},{kl_div:.4f}\n")
    
    def train(self, num_epochs: int = 5):
        """메인 학습 루프"""
        logger.info(f"🚀 Simple GRPO 학습 시작 ({num_epochs} 에포크)")
        logger.info("=" * 60)
        
        training_prompts = get_training_prompts()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            logger.info(f"\n🎯 에포크 {epoch + 1}/{num_epochs}")
            
            try:
                # 롤아웃 수집 (이미지 저장 포함)
                experiences = self.collect_rollouts_with_logging(training_prompts, epoch + 1)
                
                if not experiences:
                    logger.warning(f"⚠️ 에포크 {epoch + 1}: 경험 없음, 건너뛰기")
                    continue
                
                # 정책 업데이트
                metrics = self.update_policy(experiences)
                
                # 에포크 시간 기록
                epoch_time = time.time() - epoch_start_time
                self.training_metrics['epoch_times'].append(epoch_time)
                
                # 상세 메트릭 로깅
                self.log_epoch_metrics(epoch + 1, experiences, metrics)
                
                # 플롯 생성 (매 에포크마다)
                self.plot_training_metrics(epoch + 1)
                
                # GPU 메모리 정리
                if self.use_gpu:
                    for gpu_id in range(min(3, torch.cuda.device_count())):
                        with torch.cuda.device(gpu_id):
                            torch.cuda.empty_cache()
                
                logger.info(f"⏱️ 에포크 {epoch + 1} 완료 시간: {epoch_time:.2f}초")
                
            except Exception as e:
                logger.error(f"❌ 에포크 {epoch + 1} 실패: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        logger.info("🎉 Simple GRPO 학습 완료!")

def main():
    """메인 함수"""
    logger.info("🚀 Simple Multi-GPU GRPO 학습 시작")
    logger.info("=" * 60)
    logger.info("GPU 할당:")
    logger.info("  GPU 0: QWEN LoRA Training")
    logger.info("  GPU 1: CLIP Reward Calculation")
    logger.info("  GPU 2: Stable Diffusion 3 Image Generation")
    logger.info("=" * 60)
    
    # 설정 - 고성능 LoRA 최적화
    config = QWENGRPOConfig(
        learning_rate=2e-4,  # 더 높은 학습률 (확장된 LoRA에 맞춤)
        batch_size=3,  # 배치 크기 증가 (메모리 여유분 활용)
        num_rollouts=3,  # 롤아웃 수 증가
        max_prompt_length=77,
        max_new_tokens=30,  # 토큰 수 증가
        temperature=1.2,
        top_p=0.9,
        top_k=100,
        kl_coef=0.02,
        clip_ratio=0.2,
        entropy_coef=0.01,
        save_images=True,
        log_dir="grpo_enhanced_results"  # 새로운 결과 디렉토리
    )
    
    try:
        # 트레이너 초기화
        trainer = SimpleGRPOTrainer(config)
        
        # 학습 실행
        trainer.train(num_epochs=3)
        
    except Exception as e:
        logger.error(f"❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # GPU 메모리 정리
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
            logger.info("🧹 GPU 메모리 정리 완료")

if __name__ == "__main__":
    main()
