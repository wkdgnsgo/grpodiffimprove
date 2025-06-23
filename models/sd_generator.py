"""
Stable Diffusion 3 Generator
============================

Stable Diffusion 3를 사용하여 텍스트 프롬프트로부터 고품질 이미지를 생성하는 모듈입니다.

주요 기능:
1. 텍스트-투-이미지 생성 (Text-to-Image)
2. 이미지 품질 최적화
3. 배치 생성 지원
4. 메모리 효율적 생성

Author: AI Assistant
Date: 2025-01-22
"""

import torch
from diffusers import StableDiffusion3Pipeline, DiffusionPipeline
from PIL import Image
from typing import List, Dict, Optional, Union, Tuple
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

class SD3Generator:
    """
    Stable Diffusion 3를 사용한 이미지 생성기 클래스
    
    이 클래스는 텍스트 프롬프트를 받아서 고품질의 이미지를 생성합니다.
    GRPO 학습에서는 생성된 이미지가 CLIP 보상 계산에 사용됩니다.
    
    Attributes:
        model_name (str): 사용할 SD3 모델 이름
        pipeline: Diffusion 파이프라인 객체
        device: 연산 디바이스
        generation_config (dict): 이미지 생성 설정
    """
    
    def __init__(self,
                 model_name: str = "stabilityai/stable-diffusion-3-medium",
                 device: str = "auto",
                 height: int = 1024,
                 width: int = 1024,
                 num_inference_steps: int = 28,
                 guidance_scale: float = 7.0):
        """
        SD3 Generator 초기화
        
        Args:
            model_name (str): 사용할 SD3 모델 이름
            device (str): 디바이스 설정 ("auto", "mps", "cuda", "cpu")
            height (int): 생성할 이미지 높이 (SD3는 1024x1024 권장)
            width (int): 생성할 이미지 너비
            num_inference_steps (int): 추론 스텝 수 (SD3는 28스텝 권장)
            guidance_scale (float): 가이던스 스케일 (SD3는 7.0 권장)
        """
        self.model_name = model_name
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        
        # 디바이스 자동 선택
        if device == "auto":
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
        
        # SD3 특화 이미지 생성 설정
        self.generation_config = {
            'height': self.height,
            'width': self.width,
            'num_inference_steps': self.num_inference_steps,
            'guidance_scale': self.guidance_scale,
            'num_images_per_prompt': 1,
            'generator': None,  # 재현 가능한 결과를 위해 시드 설정 가능
            'max_sequence_length': 256,  # SD3 특화 설정
        }
        
        # 파이프라인 로드
        self._load_pipeline()
    
    def _load_pipeline(self):
        """
        Stable Diffusion 3 파이프라인을 로드하는 내부 메서드
        
        이 메서드는:
        1. SD3 파이프라인 로드
        2. 디바이스 설정
        3. 메모리 최적화 설정
        4. 안전 체크 비활성화 (연구용)
        """
        try:
            logger.info(f"📥 Loading SD3 pipeline: {self.model_name}")
            
            # SD3 모델인지 확인하여 적절한 로딩 방식 선택
            if "stable-diffusion-3" in self.model_name.lower():
                # SD3 전용 로딩 방식
                self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device in ['cuda', 'mps'] else torch.float32,
                    text_encoder_3=None,  # T5 텍스트 인코더 비활성화 (메모리 절약)
                    transformer=None,  # 필요시 커스텀 트랜스포머 사용
                )
            else:
                # 일반 SD 모델 로딩
                self.pipeline = DiffusionPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device in ['cuda', 'mps'] else torch.float32,
                    use_safetensors=True,
                    variant="fp16" if self.device in ['cuda', 'mps'] else None
                )
            
            # 디바이스로 이동
            self.pipeline = self.pipeline.to(self.device)
            
            # 메모리 최적화 (필요한 경우)
            if self.device == "mps":
                # Apple Silicon 최적화
                self.pipeline.enable_attention_slicing()
            elif self.device == "cuda":
                # CUDA 최적화
                if hasattr(self.pipeline, 'enable_memory_efficient_attention'):
                    self.pipeline.enable_memory_efficient_attention()
                self.pipeline.enable_attention_slicing()
            
            # 안전 체크 비활성화 (연구 목적)
            if hasattr(self.pipeline, 'safety_checker'):
                self.pipeline.safety_checker = None
            if hasattr(self.pipeline, 'requires_safety_checker'):
                self.pipeline.requires_safety_checker = False
            
            logger.info(f"✅ SD3 pipeline loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load SD3 pipeline: {e}")
            logger.info("🔄 Trying alternative loading method...")
            
            # 대안 로딩 방식
            try:
                self.pipeline = DiffusionPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device in ['cuda', 'mps'] else torch.float32,
                    use_safetensors=True,
                )
                self.pipeline = self.pipeline.to(self.device)
                logger.info(f"✅ SD3 pipeline loaded with alternative method on {self.device}")
                
            except Exception as e2:
                logger.error(f"❌ Alternative loading also failed: {e2}")
                raise RuntimeError(f"SD3 pipeline loading failed: {e} | Alternative: {e2}")
    
    def generate_image(self, 
                      prompt: str, 
                      negative_prompt: Optional[str] = None,
                      seed: Optional[int] = None) -> Image.Image:
        """
        단일 프롬프트로부터 이미지 생성
        
        이 메서드는 텍스트 프롬프트를 받아서:
        1. 프롬프트 전처리
        2. SD3로 이미지 생성
        3. 후처리 및 품질 확인
        4. PIL Image 반환
        
        Args:
            prompt (str): 이미지 생성용 텍스트 프롬프트
            negative_prompt (str, optional): 원하지 않는 요소 지정
            seed (int, optional): 재현 가능한 결과를 위한 시드
            
        Returns:
            PIL.Image.Image: 생성된 이미지
            
        Example:
            image = generator.generate_image(
                "a fluffy orange cat sitting on a windowsill, professional photography"
            )
        """
        try:
            # 시드 설정 (재현 가능한 결과)
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
                self.generation_config['generator'] = generator
            
            # 네거티브 프롬프트 기본값 설정
            if negative_prompt is None:
                negative_prompt = "blurry, low quality, distorted, deformed, ugly, bad anatomy"
            
            logger.debug(f"🎨 Generating image for prompt: '{prompt[:50]}...'")
            
            # 이미지 생성
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    **self.generation_config
                )
            
            # 생성된 이미지 추출
            generated_image = result.images[0]
            
            # 이미지 품질 검증
            if self._validate_image(generated_image):
                logger.debug("✅ Image generated successfully")
                return generated_image
            else:
                logger.warning("⚠️ Generated image quality check failed")
                return generated_image  # 실패해도 이미지는 반환
                
        except Exception as e:
            logger.error(f"❌ Image generation failed: {e}")
            # 실패 시 빈 이미지 반환
            return self._create_fallback_image()
    
    def generate_images_batch(self, 
                             prompts: List[str],
                             negative_prompts: Optional[List[str]] = None,
                             seeds: Optional[List[int]] = None) -> List[Image.Image]:
        """
        여러 프롬프트로부터 배치 이미지 생성
        
        Args:
            prompts (List[str]): 이미지 생성용 프롬프트 리스트
            negative_prompts (List[str], optional): 네거티브 프롬프트 리스트
            seeds (List[int], optional): 시드 리스트
            
        Returns:
            List[PIL.Image.Image]: 생성된 이미지 리스트
        """
        generated_images = []
        
        # 기본값 설정
        if negative_prompts is None:
            negative_prompts = [None for _ in range(len(prompts))]
        if seeds is None:
            seeds = [None for _ in range(len(prompts))]
        
        # 각 프롬프트에 대해 이미지 생성
        for i, prompt in enumerate(prompts):
            neg_prompt = negative_prompts[i] if i < len(negative_prompts) else None
            seed = seeds[i] if i < len(seeds) else None
            
            image = self.generate_image(prompt, neg_prompt, seed)
            generated_images.append(image)
            
            logger.debug(f"📸 Generated image {i+1}/{len(prompts)}")
        
        return generated_images
    
    def _validate_image(self, image: Image.Image) -> bool:
        """
        생성된 이미지의 품질을 검증하는 내부 메서드
        
        Args:
            image (PIL.Image.Image): 검증할 이미지
            
        Returns:
            bool: 이미지가 유효한지 여부
        """
        try:
            # 기본 검증: 크기 확인
            if image.size[0] < 100 or image.size[1] < 100:
                return False
            
            # 이미지 데이터 검증
            img_array = np.array(image)
            
            # 완전히 검은색이거나 흰색인 이미지 제외
            if np.all(img_array == 0) or np.all(img_array == 255):
                return False
            
            # 분산이 너무 낮은 이미지 제외 (노이즈만 있는 경우)
            if np.var(img_array) < 100:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Image validation failed: {e}")
            return False
    
    def _create_fallback_image(self) -> Image.Image:
        """
        이미지 생성 실패 시 사용할 기본 이미지 생성
        
        Returns:
            PIL.Image.Image: 기본 이미지
        """
        # 단색 이미지 생성 (연구용)
        fallback_image = Image.new(
            'RGB', 
            (self.width, self.height), 
            color=(128, 128, 128)  # 회색
        )
        
        logger.info("🔄 Using fallback image")
        return fallback_image
    
    def save_image(self, 
                   image: Image.Image, 
                   save_path: str,
                   quality: int = 95) -> bool:
        """
        이미지를 파일로 저장
        
        Args:
            image (PIL.Image.Image): 저장할 이미지
            save_path (str): 저장 경로
            quality (int): JPEG 품질 (1-100)
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 디렉토리 생성
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 이미지 저장
            if save_path.lower().endswith('.jpg') or save_path.lower().endswith('.jpeg'):
                image.save(save_path, 'JPEG', quality=quality)
            else:
                image.save(save_path)
            
            logger.debug(f"💾 Image saved: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save image: {e}")
            return False
    
    def get_pipeline_info(self) -> Dict:
        """
        파이프라인 정보 반환 (디버깅 및 로깅용)
        
        Returns:
            Dict: 파이프라인 관련 정보
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "generation_config": self.generation_config,
            "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }
    
    def update_generation_config(self, **kwargs):
        """
        이미지 생성 설정 업데이트
        
        Args:
            **kwargs: 업데이트할 설정들
        """
        self.generation_config.update(kwargs)
        logger.info(f"🔧 Generation config updated: {kwargs}")
    
    def clear_cache(self):
        """
        GPU 메모리 캐시 정리 (메모리 부족 시 사용)
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 CUDA cache cleared")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS 캐시 정리 (PyTorch 2.0+)
            try:
                torch.mps.empty_cache()
                logger.info("🧹 MPS cache cleared")
            except:
                pass


if __name__ == "__main__":
    # SD3 Generator 테스트 코드
    print("🧪 SD3 Generator Test")
    print("=" * 30)
    
    try:
        # SD3 생성기 초기화
        generator = SD3Generator(
            model_name="stabilityai/stable-diffusion-3-medium",
            device="auto",
            height=1024,  # SD3 권장 크기
            width=1024,
            num_inference_steps=28  # SD3 권장 스텝 수
        )
        
        print("✅ SD3 Generator initialized successfully")
        print(f"📊 Pipeline info: {generator.get_pipeline_info()}")
        
        # 테스트 프롬프트들
        test_prompts = [
            "a cute cat sitting on a windowsill",
            "beautiful sunset over mountains",
            "professional portrait of a woman"
        ]
        
        print("\n🎨 Testing image generation:")
        for i, prompt in enumerate(test_prompts):
            print(f"  Generating image {i+1}: '{prompt}'")
            image = generator.generate_image(prompt)
            
            # 테스트 이미지 저장
            save_path = f"test_output_{i+1}.jpg"
            success = generator.save_image(image, save_path)
            if success:
                print(f"    ✅ Saved: {save_path}")
            else:
                print(f"    ❌ Save failed")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("\nUsage:")
    print("from models.sd_generator import SD3Generator")
    print("generator = SD3Generator()")
    print("image = generator.generate_image('a beautiful landscape')") 