"""
Stable Diffusion 3 Generator for GRPO Training
==============================================

Stable Diffusion 3를 사용하여 향상된 프롬프트로부터 이미지를 생성하는 모듈입니다.
GRPO 학습에서 환경 역할을 담당합니다.

주요 기능:
1. Enhanced prompt로 고품질 이미지 생성
2. 배치 생성 지원
3. 메모리 효율적 관리
4. HuggingFace 토큰 자동 로그인 지원

Author: AI Assistant
Date: 2025-01-22
"""

import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from typing import List, Dict, Optional, Union, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

class SD3Generator:
    """
    Stable Diffusion 3를 사용한 이미지 생성기 클래스
    
    GRPO 학습에서 환경 역할:
    - State: user_prompt + placeholder + generated_tokens
    - Action: 다음 토큰 선택
    - Reward: CLIP(user_prompt, generated_image)
    """
    
    def __init__(self,
                 model_name: str = "stabilityai/stable-diffusion-3-medium-diffusers",
                 device: str = "auto",
                 height: int = 1024,
                 width: int = 1024,
                 num_inference_steps: int = 28,
                 guidance_scale: float = 7.0):
        """
        SD3 Generator 초기화
        
        Args:
            model_name (str): SD3 모델 이름
            device (str): 디바이스 설정
            height (int): 생성할 이미지 높이
            width (int): 생성할 이미지 너비
            num_inference_steps (int): 추론 스텝 수
            guidance_scale (float): 가이던스 스케일
        """
        self.model_name = model_name
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        
        # 디바이스 자동 선택 (Multi-GPU 지원)
        if device == "auto":
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from gpu_config import get_device_for_model
                device_str = get_device_for_model('sd3')
                self.device = device_str
                logger.info(f"🚀 Using assigned GPU for SD3: {device_str}")
            except ImportError:
                if torch.backends.mps.is_available():
                    self.device = "mps"
                    logger.info("🍎 Using Apple Silicon MPS for SD3")
                elif torch.cuda.is_available():
                    self.device = "cuda"
                    logger.info("🚀 Using CUDA GPU for SD3")
                else:
                    self.device = "cpu"
                    logger.info("💻 Using CPU for SD3 (slow)")
        else:
            self.device = device
        
        # 파이프라인 로드
        self._load_pipeline()
        
        logger.info(f"✅ SD3 Generator initialized: {self.model_name}")
    
    def _load_pipeline(self):
        """SD3 파이프라인 로드"""
        try:
            logger.info(f"📥 Loading SD3 pipeline: {self.model_name}")
            
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 
            )
            
            # 디바이스로 이동
            self.pipeline = self.pipeline.to(self.device)
            
            # 메모리 최적화
            if self.device == "mps":
                self.pipeline.enable_attention_slicing()
            elif self.device == "cuda":
                self.pipeline.enable_attention_slicing()
                self.pipeline.enable_model_cpu_offload()
            
            # 안전 체크 비활성화 (연구용)
            if hasattr(self.pipeline, 'safety_checker'):
                self.pipeline.safety_checker = None
            if hasattr(self.pipeline, 'requires_safety_checker'):
                self.pipeline.requires_safety_checker = False
            
            # Progress bar 비활성화
            self.pipeline.set_progress_bar_config(disable=True)
            
            logger.info(f"✅ SD3 pipeline loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load SD3 pipeline: {e}")
            raise RuntimeError(f"SD3 pipeline loading failed: {e}")
    
    def generate_image(self, prompt: str, seed: Optional[int] = None) -> Optional[Image.Image]:
        """
        프롬프트로 이미지 생성
        
        Args:
            prompt (str): 이미지 생성 프롬프트
            seed (int, optional): 재현 가능한 결과를 위한 시드
            
        Returns:
            PIL.Image.Image: 생성된 이미지
        """
        try:
            # 시드 설정
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # 이미지 생성
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    height=self.height,
                    width=self.width,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    generator=generator,
                    num_images_per_prompt=1
                )
            
            image = result.images[0]
            
            # 이미지 검증
            if self._validate_image(image):
                logger.debug(f"🖼️ Generated image for: '{prompt[:50]}...'")
                return image
            else:
                logger.warning(f"⚠️ Generated invalid image for: '{prompt[:50]}...'")
                return self._create_fallback_image()
                
        except Exception as e:
            logger.error(f"❌ Image generation failed: {e}")
            return self._create_fallback_image()
    
    def generate_images_batch(self, 
                             prompts: List[str],
                             seeds: Optional[List[int]] = None) -> List[Image.Image]:
        """
        배치로 여러 이미지 생성
        
        Args:
            prompts (List[str]): 프롬프트 리스트
            seeds (List[int], optional): 시드 리스트
            
        Returns:
            List[Image.Image]: 생성된 이미지들
        """
        images = []
        
        for i, prompt in enumerate(prompts):
            seed = seeds[i] if seeds and i < len(seeds) else None
            image = self.generate_image(prompt, seed)
            images.append(image)
        
        logger.info(f"📊 Generated {len(images)} images in batch")
        return images
    
    def _validate_image(self, image: Image.Image) -> bool:
        """생성된 이미지 검증"""
        try:
            if image is None:
                return False
            
            # 크기 검증
            if image.size[0] < 64 or image.size[1] < 64:
                return False
            
            # 채널 검증
            if len(image.getbands()) < 3:
                return False
            
            # 픽셀 값 검증 (모든 픽셀이 동일한 색이 아닌지)
            img_array = np.array(image)
            if np.std(img_array) < 1.0:  # 너무 단조로운 이미지
                return False
            
            return True
            
        except Exception:
            return False
    
    def _create_fallback_image(self) -> Image.Image:
        """실패 시 폴백 이미지 생성"""
        # 단순한 그라디언트 이미지 생성
        width, height = self.width, self.height
        image = Image.new('RGB', (width, height))
        
        pixels = []
        for y in range(height):
            for x in range(width):
                # 간단한 그라디언트
                r = int(255 * x / width)
                g = int(255 * y / height)
                b = 128
                pixels.append((r, g, b))
        
        image.putdata(pixels)
        return image
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'image_size': f"{self.width}x{self.height}",
            'inference_steps': self.num_inference_steps,
            'guidance_scale': self.guidance_scale
        } 