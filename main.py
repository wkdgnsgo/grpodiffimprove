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
        """각 GPU별 모델 초기화"""
        logger.info("🔧 모델들 초기화 중...")
        
        # GPU 0: QWEN LoRA 모델
        qwen_device = "cuda:0" if self.use_gpu else "cpu"
        logger.info(f"🧠 {qwen_device}: QWEN LoRA 모델 로딩...")
        self.qwen_model = QWENModel(
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            device=qwen_device,
            temperature=0.7,
            grpo_config=self.config,
            is_main_process=True
        )
        logger.info(f"✅ QWEN 모델 로드 완료 ({qwen_device})")
        
        # GPU 1: CLIP 리워드 모델  
        clip_device = "cuda:1" if self.use_gpu and torch.cuda.device_count() > 1 else "cuda:0" if self.use_gpu else "cpu"
        logger.info(f"🎯 {clip_device}: CLIP 리워드 모델 로딩...")
        self.reward_model = CLIPReward(device=clip_device)
        logger.info(f"✅ CLIP 모델 로드 완료 ({clip_device})")
        
        # GPU 2: Stable Diffusion 3
        sd_device = "cuda:2" if self.use_gpu and torch.cuda.device_count() > 2 else "cuda:0" if self.use_gpu else "cpu"
        logger.info(f"🎨 {sd_device}: SD3 파이프라인 로딩...")
        self.sd_pipeline = load_stable_diffusion_pipeline(device=sd_device)
        logger.info(f"✅ SD3 파이프라인 로드 완료 ({sd_device})")
        
        logger.info("🎯 모든 모델 초기화 완료!")
        
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
        """이미지 생성 (GPU 2에서 실행)"""
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
    
    def calculate_rewards(self, user_prompt: str, enhanced_prompts: List[str], images: List) -> List[float]:
        """리워드 계산 (GPU 1에서 실행)"""
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
    
    def collect_rollouts(self, prompts: List[str]) -> List[Dict]:
        """롤아웃 수집"""
        all_experiences = []
        
        for prompt_idx, user_prompt in enumerate(prompts):
            logger.info(f"\n📝 프롬프트 {prompt_idx + 1}/{len(prompts)}: '{user_prompt}'")
            
            # 1단계: 향상된 프롬프트 생성 (GPU 0)
            enhanced_data = self.generate_enhanced_prompts(user_prompt, self.config.num_rollouts)
            
            if not enhanced_data:
                logger.warning(f"⚠️ 프롬프트 생성 실패, 건너뛰기")
                continue
            
            enhanced_prompts = [data[0] for data in enhanced_data]
            log_probs = [data[1] for data in enhanced_data]
            
            # 2단계: 이미지 생성 (GPU 2)  
            images = self.generate_images(enhanced_prompts)
            
            # 3단계: 리워드 계산 (GPU 1)
            rewards = self.calculate_rewards(user_prompt, enhanced_prompts, images)
            
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
                        'original_reward': reward,
                        'enhanced_reward': reward
                    }
                }
                all_experiences.append(experience)
        
        logger.info(f"\n📊 총 수집된 경험: {len(all_experiences)}개")
        return all_experiences
    
    def update_policy(self, experiences: List[Dict]) -> Dict:
        """정책 업데이트 (GPU 0에서 실행)"""
        logger.info(f"🔄 정책 업데이트 ({len(experiences)}개 경험)")
        
        metrics = self.qwen_model.update_grpo_policy(experiences)
        
        logger.info(f"✅ 정책 업데이트 완료")
        return metrics
    
    def train(self, num_epochs: int = 5):
        """메인 학습 루프"""
        logger.info(f"🚀 Simple GRPO 학습 시작 ({num_epochs} 에포크)")
        logger.info("=" * 60)
        
        training_prompts = get_training_prompts()
        
        for epoch in range(num_epochs):
            logger.info(f"\n🎯 에포크 {epoch + 1}/{num_epochs}")
            
            try:
                # 롤아웃 수집
                experiences = self.collect_rollouts(training_prompts)
                
                if not experiences:
                    logger.warning(f"⚠️ 에포크 {epoch + 1}: 경험 없음, 건너뛰기")
                    continue
                
                # 정책 업데이트
                metrics = self.update_policy(experiences)
                
                # 메트릭 로깅
                avg_reward = sum(exp['reward'] for exp in experiences) / len(experiences)
                logger.info(f"📊 에포크 {epoch + 1} 결과:")
                logger.info(f"  - 평균 리워드: {avg_reward:.4f}")
                logger.info(f"  - 경험 수: {len(experiences)}")
                
                if metrics:
                    for key, value in metrics.items():
                        logger.info(f"  - {key}: {value:.4f}")
                
                # GPU 메모리 정리
                if self.use_gpu:
                    for gpu_id in range(min(3, torch.cuda.device_count())):
                        with torch.cuda.device(gpu_id):
                            torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"❌ 에포크 {epoch + 1} 실패: {e}")
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
    
    # 설정
    config = QWENGRPOConfig(
        learning_rate=5e-7,
        batch_size=2,
        num_rollouts=2,
        max_prompt_length=77,
        max_new_tokens=25,
        temperature=1.2,
        top_p=0.9,
        top_k=100,
        kl_coef=0.02,
        clip_ratio=0.2,
        entropy_coef=0.01,
        save_images=True,
        log_dir="simple_grpo_results"
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
