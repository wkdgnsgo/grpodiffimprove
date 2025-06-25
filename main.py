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
        # episodes 폴더만 생성 (이미지와 플롯 모두 여기에 저장)
        os.makedirs(os.path.join(self.log_dir, "episodes"), exist_ok=True)
        
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
        
        # GPU 0: QWEN LoRA 모델만 - 7B 모델 사용
        qwen_device = "cuda:0" if self.use_gpu else "cpu"
        logger.info(f"🧠 {qwen_device}: QWEN 7B LoRA 모델 로딩...")
        self.qwen_model = QWENModel(
            model_name="Qwen/Qwen2-VL-7B-Instruct",  # 2B → 7B로 변경
            device=qwen_device,
            temperature=0.7,
            grpo_config=self.config
        )
        logger.info(f"✅ QWEN 7B 모델 로드 완료 ({qwen_device})")
        
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
        logger.info(f"  GPU 0: QWEN 7B LoRA 모델 ({qwen_device})")
        logger.info(f"  GPU 1: CLIP 리워드 모델 + Reference 7B 모델 ({clip_device})")
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
    
    def calculate_batch_clip_rewards(self, user_prompt: str, enhanced_prompts: List[str], images: List) -> List[float]:
        """배치로 CLIP 리워드 계산"""
        if not images or self.reward_model is None:
            return [0.0] * len(enhanced_prompts)
        
        try:
            # 배치로 리워드 계산
            rewards = []
            for i, image in enumerate(images):
                enhanced_prompt = enhanced_prompts[i] if i < len(enhanced_prompts) else user_prompt
                reward = self.reward_model.calculate_reward(user_prompt, enhanced_prompt, image)
                rewards.append(reward)
            
            return rewards
            
        except Exception as e:
            logger.error(f"❌ 배치 CLIP 리워드 계산 실패: {e}")
            return [0.0] * len(enhanced_prompts)
    
    def calculate_detailed_rewards_batch(self, user_prompt: str, enhanced_prompts: List[str], images: List) -> List[dict]:
        """배치로 상세 리워드 계산 (CLIP, Aesthetic, Semantic 구분)"""
        if not images or self.reward_model is None:
            return [{'total_reward': 0.0, 'clip_score': 0.0, 'aesthetic_score': 0.0, 
                    'semantic_similarity': 0.0, 'clip_penalty': 0.0, 'semantic_penalty': 0.0}] * len(enhanced_prompts)
        
        try:
            detailed_rewards = []
            for i, (enhanced_prompt, image) in enumerate(zip(enhanced_prompts, images)):
                # CLIPReward.calculate_reward()를 사용하여 전체 리워드 계산
                total_reward = self.reward_model.calculate_reward(user_prompt, enhanced_prompt, image)
                
                # 개별 구성 요소 계산 (분석용)
                # 1. CLIP 유사도 직접 계산
                inputs = self.reward_model.processor(
                    text=[user_prompt], 
                    images=[image], 
                    return_tensors="pt", 
                    padding=True
                ).to(self.reward_model.device)
                
                with torch.no_grad():
                    outputs = self.reward_model.model(**inputs)
                    image_embeds = outputs.image_embeds
                    text_embeds = outputs.text_embeds
                    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                    clip_similarity = torch.sum(text_embeds * image_embeds, dim=-1).item()
                
                # 2. Aesthetic score 직접 계산
                pixel_values = (
                    self.reward_model.aesthetic_predictor_preprocessor(images=image, return_tensors="pt")
                    .pixel_values.to(torch.float16).to(self.reward_model.device)
                )
                with torch.inference_mode():
                    aesthetic_score = self.reward_model.aesthetic_predictor_model(pixel_values).logits.squeeze().item()
                
                # 3. Semantic similarity 직접 계산
                original_embed = self.reward_model.sentence_transformer_model.encode(user_prompt, convert_to_tensor=True)
                enhanced_embed = self.reward_model.sentence_transformer_model.encode(enhanced_prompt, convert_to_tensor=True)
                from sentence_transformers import util
                semantic_similarity = util.cos_sim(original_embed, enhanced_embed).item()
                
                # 4. 패널티 계산 (CLIPReward와 동일한 방식)
                clip_pen_threshold = 0.28
                aesthetic_weight_factor = 20.0
                semantic_sim_threshold = 0.7
                
                clip_penalty = aesthetic_weight_factor * min(clip_similarity - clip_pen_threshold, 0)
                semantic_penalty = -10 * (semantic_sim_threshold - semantic_similarity) if semantic_similarity < semantic_sim_threshold else 0.0
                
                detailed_reward = {
                    'total_reward': total_reward,  # CLIPReward에서 계산된 값 그대로 사용
                    'clip_score': clip_similarity,  # CLIP 유사도 (0~1)
                    'aesthetic_score': aesthetic_score,  # 미적 점수
                    'semantic_similarity': semantic_similarity,  # 의미적 유사도 (0~1)
                    'clip_penalty': clip_penalty,  # CLIP 패널티 (음수 또는 0)
                    'semantic_penalty': semantic_penalty  # 의미적 패널티 (음수 또는 0)
                }
                detailed_rewards.append(detailed_reward)
            
            return detailed_rewards
            
        except Exception as e:
            logger.error(f"❌ 상세 리워드 계산 실패: {e}")
            return [{'total_reward': 0.0, 'clip_score': 0.0, 'aesthetic_score': 0.0, 
                    'semantic_similarity': 0.0, 'clip_penalty': 0.0, 'semantic_penalty': 0.0}] * len(enhanced_prompts)
    
    def calculate_aesthetic_score(self, image) -> float:
        """이미지 미적 품질 점수 계산 (간단한 휴리스틱)"""
        try:
            # PIL Image를 numpy array로 변환
            import numpy as np
            img_array = np.array(image)
            
            # 색상 다양성 계산
            if len(img_array.shape) == 3:
                color_variance = np.var(img_array, axis=(0, 1)).mean()
                color_diversity = min(1.0, color_variance / 1000.0)
            else:
                color_diversity = 0.5
            
            # 대비 계산
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            contrast = np.std(gray) / 255.0
            
            # 전체적인 밝기 분포
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2  # 중간 밝기가 최적
            
            # 종합 점수
            aesthetic_score = (color_diversity * 0.4 + contrast * 0.4 + brightness_score * 0.2)
            return min(1.0, max(0.0, aesthetic_score))
            
        except Exception as e:
            logger.warning(f"⚠️ 미적 점수 계산 실패: {e}")
            return 0.5  # 기본값
    
    def calculate_semantic_similarity(self, user_prompt: str, enhanced_prompt: str, image) -> float:
        """의미적 유사성 계산"""
        try:
            # CLIP을 사용해서 원본 프롬프트와 향상된 프롬프트의 이미지 유사성 비교
            if self.reward_model is None:
                return 0.5
            
            original_similarity = self.reward_model.calculate_reward(user_prompt, user_prompt, image)
            enhanced_similarity = self.reward_model.calculate_reward(enhanced_prompt, enhanced_prompt, image)
            
            # 원본 프롬프트와의 유사성을 기준으로 계산
            # 향상된 프롬프트가 원본 의미를 유지하면서 개선되었는지 평가
            semantic_score = original_similarity * 0.7 + enhanced_similarity * 0.3
            
            return min(1.0, max(0.0, semantic_score))
            
        except Exception as e:
            logger.warning(f"⚠️ 의미적 유사성 계산 실패: {e}")
            return 0.5  # 기본값
    
    def make_safe_filename(self, text: str, max_length: int = 50) -> str:
        """텍스트를 안전한 파일명으로 변환"""
        import re
        # 특수문자 제거 및 공백을 언더스코어로 변경
        safe_text = re.sub(r'[<>:"/\\|?*]', '', text)
        safe_text = re.sub(r'\s+', '_', safe_text)
        safe_text = safe_text.strip('_')
        
        # 길이 제한
        if len(safe_text) > max_length:
            safe_text = safe_text[:max_length].rstrip('_')
        
        # 빈 문자열 방지
        if not safe_text:
            safe_text = "unknown_prompt"
        
        return safe_text
    
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
        """에피소드별 상세 이미지 및 비교 분석 저장"""
        try:
            # 원본 프롬프트별 폴더 생성 (안전한 폴더명으로 변환)
            safe_prompt = self.make_safe_filename(user_prompt)
            prompt_dir = os.path.join(self.log_dir, "episodes", safe_prompt)
            os.makedirs(prompt_dir, exist_ok=True)  # 프롬프트별 하위 폴더는 필요시에만 생성
            
            # 1. 원본 프롬프트로 이미지 생성 (비교용)
            logger.info(f"🔍 원본 프롬프트로 비교 이미지 생성: '{user_prompt}'")
            original_images = self.generate_images_batch([user_prompt])
            original_rewards = self.calculate_detailed_rewards_batch(user_prompt, [user_prompt], original_images)
            
            # 2. 향상된 프롬프트 이미지들과 함께 저장
            for i, (enhanced_prompt, enhanced_image, enhanced_reward) in enumerate(zip(enhanced_prompts, images, rewards)):
                # 이미지 저장
                original_image = original_images[0] if original_images else None
                original_reward = original_rewards[0] if original_rewards else {}
                enhanced_reward_detailed = self.calculate_detailed_rewards_batch(user_prompt, [enhanced_prompt], [enhanced_image])[0]
                
                # 원본 이미지 저장
                if original_image:
                    original_path = os.path.join(prompt_dir, f"epoch_{epoch}_sample_{i}_original.png")
                    original_image.save(original_path)
                
                # 향상된 이미지 저장
                enhanced_path = os.path.join(prompt_dir, f"epoch_{epoch}_sample_{i}_enhanced.png")
                enhanced_image.save(enhanced_path)
                
                # 상세 분석 텍스트 저장
                analysis_path = os.path.join(prompt_dir, f"epoch_{epoch}_sample_{i}_analysis.txt")
                with open(analysis_path, 'w', encoding='utf-8') as f:
                    f.write("=" * 60 + "\n")
                    f.write(f"EPISODE {epoch} - SAMPLE {i} ANALYSIS\n")
                    f.write("=" * 60 + "\n\n")
                    
                    f.write("📝 PROMPTS:\n")
                    f.write(f"Original Prompt: {user_prompt}\n")
                    f.write(f"Enhanced Prompt: {enhanced_prompt}\n\n")
                    
                    f.write("🎨 IMAGES:\n")
                    f.write(f"Original Image: epoch_{epoch}_sample_{i}_original.png\n")
                    f.write(f"Enhanced Image: epoch_{epoch}_sample_{i}_enhanced.png\n\n")
                    
                    f.write("📊 REWARD ANALYSIS:\n")
                    f.write("-" * 40 + "\n")
                    
                    # 원본 리워드 분석
                    if original_reward:
                        f.write("Original Prompt → Generated Image:\n")
                        f.write(f"  Total Reward: {original_reward.get('total_reward', 0.0):.4f}\n")
                        f.write(f"  CLIP Score: {original_reward.get('clip_score', 0.0):.4f}\n")
                        f.write(f"  Aesthetic Score: {original_reward.get('aesthetic_score', 0.0):.4f}\n")
                        f.write(f"  Semantic Similarity: {original_reward.get('semantic_similarity', 0.0):.4f}\n")
                        f.write(f"  CLIP Penalty: {original_reward.get('clip_penalty', 0.0):.4f}\n")
                        f.write(f"  Semantic Penalty: {original_reward.get('semantic_penalty', 0.0):.4f}\n\n")
                    
                    # 향상된 리워드 분석
                    f.write("Enhanced Prompt → Generated Image:\n")
                    f.write(f"  Total Reward: {enhanced_reward_detailed.get('total_reward', 0.0):.4f}\n")
                    f.write(f"  CLIP Score: {enhanced_reward_detailed.get('clip_score', 0.0):.4f}\n")
                    f.write(f"  Aesthetic Score: {enhanced_reward_detailed.get('aesthetic_score', 0.0):.4f}\n")
                    f.write(f"  Semantic Similarity: {enhanced_reward_detailed.get('semantic_similarity', 0.0):.4f}\n")
                    f.write(f"  CLIP Penalty: {enhanced_reward_detailed.get('clip_penalty', 0.0):.4f}\n")
                    f.write(f"  Semantic Penalty: {enhanced_reward_detailed.get('semantic_penalty', 0.0):.4f}\n\n")
                    
                    # 개선도 분석
                    if original_reward:
                        f.write("🚀 IMPROVEMENT ANALYSIS:\n")
                        f.write("-" * 40 + "\n")
                        total_improvement = enhanced_reward_detailed.get('total_reward', 0.0) - original_reward.get('total_reward', 0.0)
                        clip_improvement = enhanced_reward_detailed.get('clip_score', 0.0) - original_reward.get('clip_score', 0.0)
                        aesthetic_improvement = enhanced_reward_detailed.get('aesthetic_score', 0.0) - original_reward.get('aesthetic_score', 0.0)
                        semantic_improvement = enhanced_reward_detailed.get('semantic_similarity', 0.0) - original_reward.get('semantic_similarity', 0.0)
                        
                        f.write(f"Total Reward Change: {total_improvement:+.4f}\n")
                        f.write(f"CLIP Score Change: {clip_improvement:+.4f}\n")
                        f.write(f"Aesthetic Score Change: {aesthetic_improvement:+.4f}\n")
                        f.write(f"Semantic Similarity Change: {semantic_improvement:+.4f}\n\n")
                        
                        # 개선 요약
                        if total_improvement > 0:
                            f.write("✅ ENHANCEMENT SUCCESSFUL - Overall improvement achieved\n")
                        else:
                            f.write("❌ ENHANCEMENT FAILED - Overall degradation occurred\n")
                    
                    f.write(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Epoch: {epoch}\n")
            
            logger.info(f"💾 에피소드 {epoch} 상세 분석 저장 완료: {len(images)}개 샘플")
            
        except Exception as e:
            logger.error(f"❌ 에피소드 이미지 저장 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_training_metrics(self, epoch: int):
        """학습 메트릭 플롯 생성 - episodes/ 폴더에 지속 업데이트"""
        try:
            # episodes 폴더 (이미 __init__에서 생성됨)
            episodes_dir = os.path.join(self.log_dir, "episodes")
            
            if not self.training_metrics['epoch_rewards']:
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'GRPO Training Progress - Updated at Epoch {epoch}', fontsize=16, fontweight='bold')
            
            # 1. 리워드 추이 (트렌드 라인 추가)
            epochs_list = list(range(1, len(self.training_metrics['epoch_rewards']) + 1))
            axes[0, 0].plot(epochs_list, self.training_metrics['epoch_rewards'], 'b-o', linewidth=2, markersize=6)
            
            # 트렌드 라인 추가
            if len(epochs_list) > 1:
                z = np.polyfit(epochs_list, self.training_metrics['epoch_rewards'], 1)
                p = np.poly1d(z)
                axes[0, 0].plot(epochs_list, p(epochs_list), "r--", alpha=0.8, 
                               label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
                axes[0, 0].legend()
            
            axes[0, 0].set_title('Average Reward Over Time', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch', fontsize=12)
            axes[0, 0].set_ylabel('Average Reward', fontsize=12)
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Policy Loss 추이
            if self.training_metrics['policy_losses']:
                axes[0, 1].plot(epochs_list, self.training_metrics['policy_losses'], 'r-o', linewidth=2, markersize=6)
                axes[0, 1].set_title('Policy Loss Over Time', fontsize=14, fontweight='bold')
                axes[0, 1].set_xlabel('Epoch', fontsize=12)
                axes[0, 1].set_ylabel('Policy Loss', fontsize=12)
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. KL Divergence 추이
            if self.training_metrics['kl_divergences']:
                axes[1, 0].plot(epochs_list, self.training_metrics['kl_divergences'], 'g-o', linewidth=2, markersize=6)
                axes[1, 0].set_title('KL Divergence Over Time', fontsize=14, fontweight='bold')
                axes[1, 0].set_xlabel('Epoch', fontsize=12)
                axes[1, 0].set_ylabel('KL Divergence', fontsize=12)
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Advantage 분포 (최근 에피소드)
            if self.training_metrics['advantages']:
                recent_advantages = self.training_metrics['advantages'][-50:]  # 최근 50개
                axes[1, 1].hist(recent_advantages, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[1, 1].axvline(np.mean(recent_advantages), color='red', linestyle='--', 
                                  label=f'Mean: {np.mean(recent_advantages):.3f}')
                axes[1, 1].legend()
                axes[1, 1].set_title('Recent Advantage Distribution', fontsize=14, fontweight='bold')
                axes[1, 1].set_xlabel('Advantage Value', fontsize=12)
                axes[1, 1].set_ylabel('Frequency', fontsize=12)
                axes[1, 1].grid(True, alpha=0.3)
            else:
                # Advantage가 없으면 리워드 히스토리 바 차트
                axes[1, 1].bar(epochs_list, self.training_metrics['epoch_rewards'], 
                              alpha=0.7, color='lightgreen', edgecolor='black')
                axes[1, 1].set_title('Reward History (Bar Chart)', fontsize=14, fontweight='bold')
                axes[1, 1].set_xlabel('Epoch', fontsize=12)
                axes[1, 1].set_ylabel('Reward', fontsize=12)
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # episodes 폴더에 메인 플롯 저장 (항상 같은 파일명으로 업데이트)
            main_plot_path = os.path.join(episodes_dir, "training_progress.png")
            plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
            
            # 에포크별 백업도 저장
            backup_plot_path = os.path.join(episodes_dir, f"training_progress_epoch_{epoch}.png")
            plt.savefig(backup_plot_path, dpi=300, bbox_inches='tight')
            
            plt.close()
            
            logger.info(f"📊 학습 진행 플롯 업데이트: {main_plot_path}")
            logger.info(f"📊 에포크 백업 저장: {backup_plot_path}")
            
        except Exception as e:
            logger.error(f"❌ 플롯 생성 실패: {e}")
            import traceback
            traceback.print_exc()
    
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
                
                # 정책 업데이트 (그라디언트 업데이트 포함)
                try:
                    logger.info(f"🔄 에포크 {epoch + 1}: 정책 업데이트 시작...")
                    metrics = self.update_policy(experiences)
                    logger.info(f"✅ 에포크 {epoch + 1}: 정책 업데이트 완료")
                    
                except Exception as update_error:
                    logger.error(f"💥 에포크 {epoch + 1}: 그라디언트 업데이트 중 치명적 오류 발생!")
                    logger.error(f"🚨 오류 내용: {update_error}")
                    import traceback
                    traceback.print_exc()
                    
                    # 모델 상태 저장 시도
                    try:
                        emergency_save_path = os.path.join(self.log_dir, f"emergency_save_epoch_{epoch + 1}")
                        os.makedirs(emergency_save_path, exist_ok=True)
                        self.qwen_model.save_lora_model(emergency_save_path)
                        logger.info(f"💾 응급 모델 저장 완료: {emergency_save_path}")
                    except Exception as save_error:
                        logger.error(f"❌ 응급 모델 저장 실패: {save_error}")
                    
                    # 학습 종료
                    logger.error(f"🛑 그라디언트 업데이트 오류로 인한 학습 조기 종료 (에포크 {epoch + 1}/{num_epochs})")
                    logger.info(f"📊 완료된 에포크: {epoch}/{num_epochs}")
                    break  # 학습 루프 종료
                
                # 에포크 시간 기록
                epoch_time = time.time() - epoch_start_time
                self.training_metrics['epoch_times'].append(epoch_time)
                
                # 상세 메트릭 로깅
                self.log_epoch_metrics(epoch + 1, experiences, metrics)
                
                # 플롯 생성 (매 에포크마다)
                self.plot_training_metrics(epoch + 1)
                
                # 적극적인 GPU 메모리 정리
                if self.use_gpu:
                    # 모든 GPU에 대해 메모리 정리
                    for gpu_id in range(torch.cuda.device_count()):
                        with torch.cuda.device(gpu_id):
                            torch.cuda.empty_cache()
                    
                    # 추가 메모리 정리 시도
                    import gc
                    gc.collect()
                    torch.cuda.synchronize()
                
                logger.info(f"⏱️ 에포크 {epoch + 1} 완료 시간: {epoch_time:.2f}초")
                
            except Exception as e:
                logger.error(f"❌ 에포크 {epoch + 1} 일반 오류: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 최종 모델 저장
        try:
            final_save_path = os.path.join(self.log_dir, "final_model")
            os.makedirs(final_save_path, exist_ok=True)
            self.qwen_model.save_lora_model(final_save_path)
            logger.info(f"💾 최종 모델 저장 완료: {final_save_path}")
        except Exception as save_error:
            logger.error(f"❌ 최종 모델 저장 실패: {save_error}")
        
        # 학습 완료 요약
        total_epochs_completed = len(self.training_metrics['epoch_rewards'])
        total_training_time = sum(self.training_metrics['epoch_times']) if self.training_metrics['epoch_times'] else 0
        
        logger.info("🎉 Simple GRPO 학습 완료!")
        logger.info("=" * 60)
        logger.info("📊 학습 요약:")
        logger.info(f"  완료된 에포크: {total_epochs_completed}/{num_epochs}")
        logger.info(f"  총 학습 시간: {total_training_time:.2f}초 ({total_training_time/60:.1f}분)")
        
        if self.training_metrics['epoch_rewards']:
            avg_reward = sum(self.training_metrics['epoch_rewards']) / len(self.training_metrics['epoch_rewards'])
            best_reward = max(self.training_metrics['epoch_rewards'])
            logger.info(f"  평균 리워드: {avg_reward:.4f}")
            logger.info(f"  최고 리워드: {best_reward:.4f}")
        
        logger.info(f"  결과 저장 위치: {self.log_dir}")
        logger.info("=" * 60)

def main():
    """메인 함수"""
    logger.info("🚀 Simple Multi-GPU GRPO 학습 시작")
    logger.info("=" * 60)
    logger.info("GPU 할당:")
    logger.info("  GPU 0: QWEN LoRA Training")
    logger.info("  GPU 1: CLIP Reward Calculation")
    logger.info("  GPU 2: Stable Diffusion 3 Image Generation")
    logger.info("=" * 60)
    
    # 설정 - QWEN 7B + 극한 메모리 최적화
    config = QWENGRPOConfig(
        learning_rate=1e-4,  # 7B 모델에 적합한 학습률
        batch_size=1,  # 극한 메모리 절약 (2 → 1)
        num_rollouts=1,  # 극한 메모리 절약 (2 → 1)
        max_prompt_length=77,
        max_new_tokens=20,  # 토큰 수 더 제한 (30 → 20)
        temperature=1.2,
        top_p=0.9,
        top_k=100,
        kl_coef=0.02,
        clip_ratio=0.2,
        entropy_coef=0.01,
        save_images=True,
        log_dir="grpo_7b_results"  # 7B 모델 전용 결과 디렉토리
    )
    
    try:
        # 트레이너 초기화
        trainer = SimpleGRPOTrainer(config)
        
        # 학습 실행
        trainer.train(num_epochs=15)
        
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
