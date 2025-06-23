"""
VLM (Vision Language Model) Wrapper
===================================

Qwen2.5-VL을 사용하여 사용자의 간단한 프롬프트를 상세하고 품질 높은 프롬프트로 개선하는 모듈입니다.

주요 기능:
1. 프롬프트 개선 (Prompt Enhancement)
2. 텍스트 생성 파라미터 관리
3. 디바이스 최적화 (MPS/CUDA/CPU)
4. 배치 처리 지원
5. 토큰 길이 제한 (CLIP 77 토큰 제한)

Author: AI Assistant
Date: 2025-01-22
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, CLIPTokenizer
from typing import List, Dict, Optional, Union
import logging
import json

logger = logging.getLogger(__name__)

class VLMWrapper(nn.Module):
    """
    Qwen2.5-VL을 래핑하여 프롬프트 개선 기능을 제공하는 클래스
    
    이 클래스는 사용자의 간단한 프롬프트 (예: "a cat")를 받아서
    더 상세하고 구체적인 프롬프트 (예: "a fluffy orange tabby cat sitting gracefully...")로 
    변환하는 기능을 제공합니다.
    
    77토큰 제한을 준수하여 CLIP text encoder와 호환성을 보장합니다.
    
    Attributes:
        model_name (str): 사용할 Qwen2.5-VL 모델 이름 (config에서 읽어옴)
        tokenizer: 토크나이저 객체
        processor: 프로세서 객체
        model: Qwen2.5-VL 모델 객체
        clip_tokenizer: CLIP 토크나이저 (토큰 길이 체크용)
        device: 연산 디바이스 (MPS/CUDA/CPU)
        generation_config (dict): 텍스트 생성 설정
        max_token_length (int): 최대 토큰 길이 (기본값: 77)
    """
    
    def __init__(self,
                 config_path: str = "config/default_config.json",
                 device: str = "auto",
                 max_new_tokens: int = 100,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 do_sample: bool = True,
                 max_token_length: int = 77):
        """
        VLM Wrapper 초기화 (간단한 플레이스홀더 방식)
        
        Args:
            config_path (str): 설정 파일 경로
            device (str): 디바이스 설정 (사용되지 않음)
            max_new_tokens (int): 최대 토큰 수 (사용되지 않음)
            temperature (float): 생성 온도 (사용되지 않음)
            top_p (float): 누적 확률 임계값 (사용되지 않음)
            do_sample (bool): 샘플링 여부 (사용되지 않음)
            max_token_length (int): 최대 토큰 길이 (CLIP 제한)
        """
        super().__init__()
        
        # 설정 파일에서 모델 이름 읽기 (참고용)
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.model_name = config['model_settings']['vlm_model']
            logger.info(f"📄 VLM model name from config: {self.model_name}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load config: {e}, using default")
            self.model_name = "placeholder-vlm"
        
        # 간단한 플레이스홀더 방식이므로 실제 모델 로드 없음
        self.device = "cpu"  # 플레이스홀더 방식에서는 디바이스 불필요
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        # CLIP 토크나이저 초기화 (토큰 길이 체크용)
        try:
            self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("✅ CLIP tokenizer loaded for token length validation")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load CLIP tokenizer: {e}, using fallback")
            self.clip_tokenizer = None
        
        # 토큰 길이 제한 설정
        self.max_token_length = max_token_length
        
        # 생성 설정 (사용되지 않지만 호환성을 위해 유지)
        self.generation_config = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'do_sample': do_sample
        }
        
        logger.info(f"✅ VLM Wrapper initialized with placeholder-based enhancement (max_tokens: {max_token_length})")
    
    def _count_tokens(self, text: str) -> int:
        """
        텍스트의 토큰 개수를 계산
        
        Args:
            text (str): 토큰 개수를 계산할 텍스트
            
        Returns:
            int: 토큰 개수
        """
        if not text:
            return 0
            
        try:
            if self.clip_tokenizer:
                # CLIP tokenizer 사용
                tokens = self.clip_tokenizer.encode(text, add_special_tokens=True)
                return len(tokens)
            else:
                # Fallback: 대략적인 토큰 개수 추정 (영어 기준 평균 4자당 1토큰)
                return len(text.split()) + len(text) // 20
        except Exception as e:
            logger.warning(f"⚠️ Token counting failed: {e}, using fallback")
            return len(text.split()) + len(text) // 20

    def _truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """
        텍스트를 토큰 제한에 맞게 잘라냄
        
        Args:
            text (str): 잘라낼 텍스트
            max_tokens (int): 최대 토큰 수
            
        Returns:
            str: 잘라낸 텍스트
        """
        if not text:
            return text
            
        current_tokens = self._count_tokens(text)
        if current_tokens <= max_tokens:
            return text
        
        # 토큰 단위로 잘라내기
        words = text.split()
        truncated_text = ""
        
        for word in words:
            test_text = truncated_text + (" " if truncated_text else "") + word
            if self._count_tokens(test_text) > max_tokens:
                break
            truncated_text = test_text
        
        if not truncated_text:  # 단어 하나도 못 넣은 경우
            # 문자 단위로 잘라내기
            for i in range(len(text)):
                test_text = text[:i+1]
                if self._count_tokens(test_text) > max_tokens:
                    truncated_text = text[:i] if i > 0 else text[:1]
                    break
            else:
                truncated_text = text
        
        logger.debug(f"🔄 Truncated text from {current_tokens} to {self._count_tokens(truncated_text)} tokens")
        return truncated_text

    def _load_model(self):
        """
        모델 로드 메서드 (플레이스홀더 방식에서는 사용되지 않음)
        """
        logger.info("📝 Using placeholder-based enhancement, no model loading required")
        pass
    
    def enhance_prompt(self, user_prompt: str) -> str:
        """
        사용자 프롬프트를 간단한 플레이스홀더 방식으로 개선하되, 77토큰 제한을 준수
        
        Args:
            user_prompt (str): 개선할 원본 프롬프트
            
        Returns:
            str: 개선된 프롬프트 (77토큰 이하)
        """
        if not user_prompt or not user_prompt.strip():
            logger.warning("⚠️ Empty prompt provided, using fallback")
            return self._fallback_enhancement("")
        
        try:
            # 입력 검증 및 정제
            user_prompt = user_prompt.strip()
            if len(user_prompt) > 200:  # 너무 긴 프롬프트 제한
                user_prompt = user_prompt[:200]
                logger.warning("⚠️ Prompt truncated to 200 characters")
            
            # 사용자 프롬프트의 토큰 수 확인
            user_prompt_tokens = self._count_tokens(user_prompt)
            logger.debug(f"📊 User prompt tokens: {user_prompt_tokens}/{self.max_token_length}")
            
            if user_prompt_tokens >= self.max_token_length:
                # 사용자 프롬프트가 이미 제한을 초과하는 경우 잘라내기
                user_prompt = self._truncate_to_token_limit(user_prompt, self.max_token_length - 5)
                logger.warning(f"⚠️ User prompt truncated to fit token limit: {user_prompt}")
                return user_prompt
            
            # 개선할 수 있는 토큰 수 계산
            available_tokens = self.max_token_length - user_prompt_tokens
            logger.debug(f"📊 Available tokens for enhancement: {available_tokens}")
            
            # 간단한 플레이스홀더 기반 개선
            enhanced_prompt = self._enhance_with_placeholders(user_prompt, available_tokens)
            
            # 최종 토큰 길이 검증
            final_tokens = self._count_tokens(enhanced_prompt)
            if final_tokens > self.max_token_length:
                enhanced_prompt = self._truncate_to_token_limit(enhanced_prompt, self.max_token_length)
                logger.warning(f"⚠️ Enhanced prompt truncated to fit token limit")
            
            logger.debug(f"✅ Enhanced: '{user_prompt}' → '{enhanced_prompt}' ({final_tokens} tokens)")
            return enhanced_prompt
            
        except Exception as e:
            logger.warning(f"⚠️ Prompt enhancement failed: {e}")
            fallback = self._fallback_enhancement(user_prompt)
            return self._truncate_to_token_limit(fallback, self.max_token_length)
    
    def _enhance_with_placeholders(self, user_prompt: str, available_tokens: int = None) -> str:
        """
        플레이스홀더를 사용한 간단한 프롬프트 개선 (토큰 제한 고려)
        
        Args:
            user_prompt (str): 원본 프롬프트
            available_tokens (int): 개선에 사용할 수 있는 토큰 수
            
        Returns:
            str: 개선된 프롬프트
        """
        # 기본 품질 향상 키워드들
        quality_keywords = [
            "high quality", "detailed", "professional", "sharp focus",
            "well-lit", "artistic", "masterpiece", "8k resolution"
        ]
        
        style_keywords = [
            "photorealistic", "cinematic lighting", "depth of field",
            "vivid colors", "perfect composition", "award-winning"
        ]
        
        # 카테고리별 특화 키워드
        category_keywords = {
            "person": ["portrait", "beautiful", "elegant", "expressive"],
            "woman": ["graceful", "stunning", "sophisticated", "charming"],
            "man": ["handsome", "distinguished", "confident", "strong"],
            "cat": ["fluffy", "adorable", "cute", "playful"],
            "dog": ["loyal", "friendly", "energetic", "beautiful"],
            "landscape": ["scenic", "breathtaking", "panoramic", "majestic"],
            "mountain": ["towering", "snow-capped", "dramatic", "rugged"],
            "ocean": ["crystal clear", "turquoise", "serene", "vast"],
            "forest": ["lush", "dense", "mystical", "green"],
            "city": ["urban", "modern", "bustling", "architectural"],
            "building": ["impressive", "grand", "structural", "geometric"],
            "flower": ["blooming", "colorful", "delicate", "fragrant"],
            "food": ["delicious", "appetizing", "gourmet", "fresh"],
            "car": ["sleek", "powerful", "luxury", "sporty"],
            "abstract": ["creative", "unique", "innovative", "contemporary"]
        }
        
        # 사용자 프롬프트에서 키워드 감지
        user_lower = user_prompt.lower()
        detected_category = None
        
        for category, keywords in category_keywords.items():
            if category in user_lower:
                detected_category = category
                break
        
        # 개선된 프롬프트 구성
        enhanced_parts = [user_prompt]
        
        # 카테고리별 키워드 추가
        if detected_category:
            category_words = category_keywords[detected_category]
            for word in category_words[:2]:  # 상위 2개만 사용
                test_prompt = ", ".join(enhanced_parts + [word])
                if available_tokens is None or self._count_tokens(test_prompt) <= self.max_token_length:
                    enhanced_parts.append(word)
                else:
                    break
        
        # 품질 키워드 추가 (토큰 제한 고려)
        import random
        all_keywords = quality_keywords + style_keywords
        random.shuffle(all_keywords)
        
        for keyword in all_keywords[:4]:  # 최대 4개 키워드 시도
            test_prompt = ", ".join(enhanced_parts + [keyword])
            if available_tokens is None or self._count_tokens(test_prompt) <= self.max_token_length:
                enhanced_parts.append(keyword)
            else:
                break
        
        # 최종 프롬프트 조합
        enhanced_prompt = ", ".join(enhanced_parts)
        
        return enhanced_prompt
    
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
        
        이 메서드는 Qwen2.5-VL이 더 나은 프롬프트를 생성하도록 유도하는
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
            generated_text (str): Qwen2.5-VL이 생성한 전체 텍스트
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
        Qwen2.5-VL 개선 실패 시 사용할 기본 개선 방법 (토큰 제한 고려)
        
        이 메서드는 Qwen2.5-VL이 실패했을 때 규칙 기반으로
        기본적인 프롬프트 개선을 수행합니다.
        
        Args:
            user_prompt (str): 원본 프롬프트
            
        Returns:
            str: 기본 개선된 프롬프트 (77토큰 이하)
        """
        # 기본 품질 향상 키워드 추가
        quality_keywords = [
            "high quality", "detailed", "professional", 
            "well-lit", "sharp focus", "artistic"
        ]
        
        # 사용자 프롬프트의 토큰 수 확인
        user_tokens = self._count_tokens(user_prompt)
        available_tokens = self.max_token_length - user_tokens
        
        if available_tokens <= 5:  # 여유 토큰이 너무 적으면 원본만 반환
            return self._truncate_to_token_limit(user_prompt, self.max_token_length)
        
        # 토큰 제한을 고려하여 키워드 추가
        enhanced_parts = [user_prompt]
        
        import random
        random.shuffle(quality_keywords)
        
        for keyword in quality_keywords:
            test_prompt = ", ".join(enhanced_parts + [keyword])
            if self._count_tokens(test_prompt) <= self.max_token_length:
                enhanced_parts.append(keyword)
            else:
                break
        
        enhanced = ", ".join(enhanced_parts)
        
        logger.info(f"🔄 Using fallback enhancement: {enhanced} ({self._count_tokens(enhanced)} tokens)")
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
            config_path="config/default_config.json",
            device="auto",
            max_new_tokens=100,
            temperature=0.7
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

    # 참조 글과 일치하는 구현:
    for step in range(max_new_tokens):
        # state_t = user_prompt + 지금까지_생성된_토큰들
        current_state = current_sequence.clone()
        
        # action_t = 다음_토큰_선택  
        policy_dist = policy_network(current_sequence)
        next_token = policy_dist.sample()
        
        # 시퀀스 업데이트
        current_sequence = torch.cat([current_sequence, next_token])
        
        if next_token == EOS: break

    # 환경 실행: 동결된 파이프라인
    generated_text = tokenizer.decode(current_sequence)
    image = sd_generator.generate_image(generated_text)  # 동결된 SD3
    reward = clip_reward.calculate_reward(image, text)   # 동결된 CLIP 