"""
CLIP Reward Calculator
=====================

CLIP 모델을 사용하여 원본 user prompt와 생성된 이미지 간의 유사도를 계산하는 모듈입니다.

핵심 특징:
- Enhanced prompt가 아닌 원본 user prompt만 사용
- 유사도 1.0에 가까울수록 높은 보상
- 간단하고 직관적인 보상 계산

Author: AI Assistant
Date: 2025-01-22
"""

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List, Dict, Optional, Union, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

class CLIPRewardCalculator:
    """
    CLIP을 사용하여 user prompt와 이미지 간의 유사도 기반 보상을 계산하는 클래스
    
    핵심 원리:
    1. 원본 user prompt와 이미지를 CLIP으로 인코딩
    2. 코사인 유사도 계산 (0~1 범위)
    3. 유사도가 1에 가까울수록 높은 보상
    
    주의: Enhanced prompt가 아닌 원본 user prompt만 사용!
    """
    
    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "auto"):
        """
        CLIP Reward Calculator 초기화
        
        Args:
            model_name (str): 사용할 CLIP 모델 이름
            device (str): 디바이스 설정 ("auto", "mps", "cuda", "cpu")
        """
        self.model_name = model_name
        
        # 디바이스 자동 선택 (Multi-GPU 지원)
        if device == "auto":
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from gpu_config import get_device_for_model
                device_str = get_device_for_model('clip')
                self.device = torch.device(device_str)
                logger.info(f"🚀 Using assigned GPU for CLIP: {device_str}")
            except ImportError:
                if torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                    logger.info("🍎 Using Apple Silicon MPS for CLIP")
                elif torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    logger.info("🚀 Using CUDA GPU for CLIP")
                else:
                    self.device = torch.device("cpu")
                    logger.info("💻 Using CPU for CLIP")
        else:
            self.device = torch.device(device)
        
        # 모델 로드
        self._load_model()
        
        logger.info(f"✅ CLIP Reward Calculator initialized: {self.model_name}")
    
    def _load_model(self):
        """CLIP 모델과 프로세서 로드"""
        try:
            logger.info(f"📥 Loading CLIP model: {self.model_name}")
            
            # CLIP 프로세서 로드
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
            # CLIP 모델 로드
            self.model = CLIPModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type in ['cuda', 'mps'] else torch.float32
            )
            
            # 디바이스로 이동
            self.model = self.model.to(self.device)
            self.model.eval()  # 평가 모드 설정
            
            logger.info(f"✅ CLIP model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load CLIP model: {e}")
            raise RuntimeError(f"CLIP model loading failed: {e}")
    
    def calculate_reward(self, 
                        user_prompt: str, 
                        image: Image.Image) -> float:
        """
        원본 user prompt와 이미지 간의 유사도 기반 보상 계산
        
        Args:
            user_prompt (str): 원본 사용자 프롬프트 (enhanced prompt 아님!)
            image (PIL.Image.Image): 생성된 이미지
            
        Returns:
            float: 유사도 보상 (0.0 ~ 1.0, 1.0에 가까울수록 좋음)
            
        Example:
            reward = calculator.calculate_reward("a cat", generated_image)
            # reward: 0.85 (높은 유사도)
        """
        try:
            # 입력 전처리
            inputs = self.processor(
                text=[user_prompt], 
                images=[image], 
                return_tensors="pt", 
                padding=True
            )
            
            # 디바이스로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # CLIP으로 특성 추출
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # 이미지와 텍스트 임베딩 추출
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                
                # 임베딩 정규화 (사용자 제시 방식으로 통일)
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                
                # 코사인 유사도 계산 (0~1 범위)
                similarity = torch.cosine_similarity(image_embeds, text_embeds, dim=-1)
                
                # 유사도를 0~1 범위로 변환 (cosine similarity는 -1~1 범위)
                reward = (similarity.item() + 1.0) / 2.0
                
                logger.info(f"🎯 Reward: {reward:.4f} for user prompt: '{user_prompt}'")
                
                return reward
                
        except Exception as e:
            logger.warning(f"⚠️ Reward calculation failed: {e}")
            # 실패 시 중간 보상 반환
            return 0.5
    
    def calculate_rewards_batch(self, 
                               user_prompts: List[str], 
                               images: List[Image.Image]) -> List[float]:
        """
        배치로 여러 prompt-image 쌍의 보상 계산
        
        Args:
            user_prompts (List[str]): 원본 사용자 프롬프트들
            images (List[Image.Image]): 생성된 이미지들
            
        Returns:
            List[float]: 각 쌍의 유사도 보상들
        """
        if len(user_prompts) != len(images):
            raise ValueError(f"Prompts({len(user_prompts)}) and images({len(images)}) count mismatch")
        
        rewards = []
        for prompt, image in zip(user_prompts, images):
            reward = self.calculate_reward(prompt, image)
            rewards.append(reward)
        
        logger.info(f"📊 Batch rewards calculated: avg={np.mean(rewards):.3f}, min={min(rewards):.3f}, max={max(rewards):.3f}")
        return rewards
    
    def get_detailed_similarity(self, 
                               user_prompt: str, 
                               image: Image.Image) -> Dict[str, float]:
        """
        상세한 유사도 정보 반환
        
        Returns:
            Dict with 'raw_similarity', 'reward', and 'confidence'
        """
        try:
            inputs = self.processor(
                text=[user_prompt], 
                images=[image], 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                
                # 원시 코사인 유사도 (-1~1)
                raw_similarity = torch.cosine_similarity(image_embeds, text_embeds, dim=-1).item()
                
                # 보상 (0~1)
                reward = (raw_similarity + 1.0) / 2.0
                
                # 신뢰도 (절댓값이 클수록 확신)
                confidence = abs(raw_similarity)
                
                return {
                    'raw_similarity': raw_similarity,
                    'reward': reward,
                    'confidence': confidence,
                    'user_prompt': user_prompt
                }
                
        except Exception as e:
            logger.error(f"❌ Detailed similarity calculation failed: {e}")
            return {
                'raw_similarity': 0.0,
                'reward': 0.5,
                'confidence': 0.0,
                'user_prompt': user_prompt,
                'error': str(e)
            }
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'purpose': 'User prompt to image similarity reward',
            'reward_range': '0.0 ~ 1.0 (higher is better)'
        }

# 테스트용 더미 이미지 생성 함수
def create_dummy_image(prompt: str, size: Tuple[int, int] = (256, 256)) -> Image.Image:
    """
    테스트용 더미 이미지 생성 (실제 이미지 생성 전 테스트용)
    
    Args:
        prompt (str): 프롬프트 (색상 결정용)
        size (Tuple[int, int]): 이미지 크기
        
    Returns:
        PIL.Image.Image: 더미 이미지
    """
    import random
    
    # 프롬프트 기반 색상 생성
    random.seed(hash(prompt) % 1000000)
    color = (
        random.randint(50, 255),
        random.randint(50, 255), 
        random.randint(50, 255)
    )
    
    # 단색 이미지 생성
    image = Image.new('RGB', size, color)
    
    return image 