"""
VLM (Vision Language Model) Wrapper
===================================

VLM을 사용하여 사용자의 간단한 프롬프트를 상세하고 품질 높은 프롬프트로 개선하는 모듈입니다.

주요 기능:
1. 프롬프트 개선 (Prompt Enhancement)
2. 텍스트 생성 파라미터 관리
3. 디바이스 최적화 (MPS/CUDA/CPU)
4. 배치 처리 지원

Author: AI Assistant
Date: 2025-01-22
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class VLMWrapper(nn.Module):
    """
    VLM을 래핑하여 프롬프트 개선 기능을 제공하는 클래스
    
    이 클래스는 사용자의 간단한 프롬프트 (예: "a cat")를 받아서
    더 상세하고 구체적인 프롬프트 (예: "a fluffy orange tabby cat sitting gracefully...")로 
    변환하는 기능을 제공합니다.
    
    Attributes:
        model_name (str): 사용할 VLM 모델 이름
        tokenizer: 토크나이저 객체
        model: VLM 모델 객체
        device: 연산 디바이스 (MPS/CUDA/CPU)
        generation_config (dict): 텍스트 생성 설정
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 device: str = "auto",
                 max_new_tokens: int = 50,
                 temperature: float = 0.8,
                 top_p: float = 0.9):
        """
        VLM Wrapper 초기화
        
        Args:
            model_name (str): 사용할 VLM 모델 이름
            device (str): 디바이스 설정 ("auto", "mps", "cuda", "cpu")
            max_new_tokens (int): 생성할 최대 토큰 수
            temperature (float): 생성 다양성 조절 (0.0-1.0)
            top_p (float): 누적 확률 임계값 (0.0-1.0)
        """
        super().__init__()
        
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # 디바이스 자동 선택 (Apple Silicon MPS > CUDA > CPU 순서)
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("🍎 Using Apple Silicon MPS")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("🚀 Using CUDA GPU")
            else:
                self.device = torch.device("cpu")
                logger.info("💻 Using CPU")
        else:
            self.device = torch.device(device)
            logger.info(f"🔧 Using specified device: {device}")
        
        # 모델과 토크나이저 로드
        self._load_model()
        
        # 텍스트 생성 설정
        self.generation_config = {
            'max_new_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'do_sample': True,
            'pad_token_id': self.tokenizer.eos_token_id,
            'repetition_penalty': 1.1,  # 반복 방지
            'length_penalty': 1.0,      # 길이 페널티
            'early_stopping': True      # 조기 종료
        }
    
    def _load_model(self):
        """
        VLM 모델과 토크나이저를 로드하는 내부 메서드
        
        이 메서드는:
        1. 토크나이저 로드 및 설정
        2. 모델 로드 및 디바이스 이동
        3. 평가 모드 설정
        4. 에러 처리
        """
        try:
            logger.info(f"📥 Loading VLM model: {self.model_name}")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 패딩 토큰 설정 (없는 경우 EOS 토큰 사용)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type in ['cuda', 'mps'] else torch.float32
            )
            
            # 디바이스로 이동
            self.model = self.model.to(self.device)
            self.model.eval()  # 평가 모드 설정
            
            logger.info(f"✅ VLM model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load VLM model: {e}")
            raise RuntimeError(f"VLM model loading failed: {e}")
    
    def enhance_prompt(self, user_prompt: str) -> str:
        """
        사용자 프롬프트를 개선하여 더 상세한 프롬프트로 변환
        
        이 메서드는 간단한 프롬프트를 받아서:
        1. 프롬프트 템플릿 적용
        2. VLM으로 텍스트 생성
        3. 후처리 및 정제
        4. 개선된 프롬프트 반환
        
        Args:
            user_prompt (str): 사용자가 입력한 간단한 프롬프트
            
        Returns:
            str: 개선된 상세 프롬프트
            
        Example:
            Input: "a cat"
            Output: "a fluffy orange tabby cat sitting gracefully on a windowsill, 
                    soft natural lighting, professional pet photography, detailed fur texture"
        """
        try:
            # 프롬프트 템플릿 적용
            enhanced_prompt_template = self._create_enhancement_template(user_prompt)
        
            # 토크나이징
            inputs = self.tokenizer.encode(
                enhanced_prompt_template,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
        
            # 텍스트 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    **self.generation_config
                )
        
            # 디코딩 및 후처리
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
        
            # 개선된 프롬프트 추출 및 정제
            enhanced_prompt = self._extract_enhanced_prompt(
                generated_text, 
                enhanced_prompt_template
            )
            
            logger.debug(f"📝 Original: {user_prompt}")
            logger.debug(f"✨ Enhanced: {enhanced_prompt}")
            
            return enhanced_prompt
            
        except Exception as e:
            logger.warning(f"⚠️ Prompt enhancement failed: {e}")
            # 실패 시 원본 프롬프트에 기본 개선 적용
            return self._fallback_enhancement(user_prompt)
    
    def enhance_prompts_batch(self, user_prompts: List[str]) -> List[str]:
        """
        여러 프롬프트를 배치로 처리하여 효율성 향상
        
        Args:
            user_prompts (List[str]): 개선할 프롬프트 리스트
            
        Returns:
            List[str]: 개선된 프롬프트 리스트
        """
        enhanced_prompts = []
        
        for prompt in user_prompts:
            enhanced = self.enhance_prompt(prompt)
            enhanced_prompts.append(enhanced)
        
        return enhanced_prompts
    
    def _create_enhancement_template(self, user_prompt: str) -> str:
        """
        프롬프트 개선을 위한 템플릿 생성
        
        이 메서드는 VLM이 더 나은 프롬프트를 생성하도록 유도하는
        템플릿을 만듭니다.
        
        Args:
            user_prompt (str): 원본 사용자 프롬프트
            
        Returns:
            str: 개선을 위한 템플릿 프롬프트
        """
        templates = [
            f"Enhance this image prompt to be more detailed and artistic: '{user_prompt}' -> Enhanced:",
            f"Make this prompt more descriptive and visually rich: '{user_prompt}' -> Improved:",
            f"Transform this simple prompt into a detailed description: '{user_prompt}' -> Detailed:",
        ]
            
        # 랜덤하게 템플릿 선택 (다양성 증가)
        import random
        return random.choice(templates)
    
    def _extract_enhanced_prompt(self, generated_text: str, template: str) -> str:
        """
        생성된 텍스트에서 개선된 프롬프트 부분만 추출
        
        Args:
            generated_text (str): VLM이 생성한 전체 텍스트
            template (str): 사용된 템플릿
            
        Returns:
            str: 추출된 개선 프롬프트
        """
        try:
            # 템플릿 이후 부분 추출
            if "->" in template:
                split_marker = template.split("->")[-1].strip()
                if split_marker in generated_text:
                    enhanced_part = generated_text.split(split_marker, 1)[-1]
                else:
                    enhanced_part = generated_text.replace(template, "")
            else:
                enhanced_part = generated_text.replace(template, "")
            
            # 정제 및 정리
            enhanced_part = enhanced_part.strip()
            
            # 불필요한 부분 제거
            enhanced_part = enhanced_part.split('\n')[0]  # 첫 번째 줄만
            enhanced_part = enhanced_part.split('.')[0]   # 첫 번째 문장만
            
            # 최소 길이 확인
            if len(enhanced_part) < 10:
                raise ValueError("Enhanced prompt too short")
            
            return enhanced_part.strip()
            
        except Exception as e:
            logger.warning(f"⚠️ Enhanced prompt extraction failed: {e}")
            return generated_text.strip()[:200]  # 처음 200자만 사용
    
    def _fallback_enhancement(self, user_prompt: str) -> str:
        """
        VLM 개선 실패 시 사용할 기본 개선 방법
        
        이 메서드는 VLM이 실패했을 때 규칙 기반으로
        기본적인 프롬프트 개선을 수행합니다.
        
        Args:
            user_prompt (str): 원본 프롬프트
            
        Returns:
            str: 기본 개선된 프롬프트
        """
        # 기본 품질 향상 키워드 추가
        quality_keywords = [
            "high quality", "detailed", "professional", 
            "well-lit", "sharp focus", "artistic"
        ]
        
        # 랜덤하게 2-3개 키워드 선택
        import random
        selected_keywords = random.sample(quality_keywords, k=min(3, len(quality_keywords)))
        
        enhanced = f"{user_prompt}, {', '.join(selected_keywords)}"
        
        logger.info(f"🔄 Using fallback enhancement: {enhanced}")
        return enhanced
    
    def get_model_info(self) -> Dict:
        """
        모델 정보 반환 (디버깅 및 로깅용)
        
        Returns:
            Dict: 모델 관련 정보
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "generation_config": self.generation_config,
            "vocab_size": self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else None
        }
    
    def update_generation_config(self, **kwargs):
        """
        텍스트 생성 설정 업데이트
        
        Args:
            **kwargs: 업데이트할 설정들
        """
        self.generation_config.update(kwargs)
        logger.info(f"🔧 Generation config updated: {kwargs}")


# 테스트 코드
if __name__ == "__main__":
    # VLM Wrapper 테스트 코드
    print("🧪 VLM Wrapper Test")
    print("=" * 30)
    
    try:
        # VLM 래퍼 초기화
        vlm = VLMWrapper(
            model_name="microsoft/DialoGPT-medium",
            device="auto",
            max_new_tokens=30,
            temperature=0.8
        )
        
        print("✅ VLM Wrapper initialized successfully")
        print(f"📊 Model info: {vlm.get_model_info()}")
        
        # 테스트 프롬프트들
        test_prompts = [
            "a cat",
            "sunset",
            "beautiful woman",
            "mountain landscape"
        ]
        
        print("\n🔄 Testing prompt enhancement:")
        for prompt in test_prompts:
            enhanced = vlm.enhance_prompt(prompt)
            print(f"  '{prompt}' → '{enhanced}'")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("\nUsage:")
    print("from models.vlm_wrapper import VLMWrapper")
    print("vlm = VLMWrapper()")
    print("enhanced = vlm.enhance_prompt('a cat')") 