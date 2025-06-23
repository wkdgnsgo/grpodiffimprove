"""
QWEN 7B Wrapper for Prompt Enhancement
=====================================

QWEN 7B 모델을 사용하여 User prompt를 향상된 prompt로 변환하는 모듈입니다.

주요 기능:
1. User prompt + placeholder 구조로 입력 구성
2. QWEN 7B로 향상된 prompt 생성
3. 생성된 prompt 검증 및 후처리

Author: AI Assistant
Date: 2025-01-22
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List, Dict, Optional, Union
import logging
import re
import json

logger = logging.getLogger(__name__)

class QwenWrapper:
    """
    QWEN VL 모델을 사용한 프롬프트 향상 클래스
    
    이 클래스는:
    1. User prompt를 받아서 placeholder와 함께 구성
    2. QWEN VL로 향상된 prompt 생성
    3. 생성된 결과 검증 및 정제
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
                 device: str = "auto",
                 max_new_tokens: int = 100,
                 temperature: float = 0.7):
        """
        QWEN Wrapper 초기화
        
        Args:
            model_name (str): QWEN 모델 이름
            device (str): 디바이스 설정
            max_new_tokens (int): 최대 생성 토큰 수
            temperature (float): 생성 온도
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # 디바이스 설정 (Multi-GPU 지원)
        if device == "auto":
            try:
                from gpu_config import get_device_for_model
                device_str = get_device_for_model('qwen')
                self.device = torch.device(device_str)
                logger.info(f"🚀 Using assigned GPU for QWEN: {device_str}")
            except ImportError:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    logger.info("🚀 Using CUDA GPU")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                    logger.info("🍎 Using Apple Silicon MPS")
                else:
                    self.device = torch.device("cpu")
                    logger.info("💻 Using CPU")
        else:
            self.device = torch.device(device)
        
        # 모델과 토크나이저 로드
        self._load_model()
        
        # 프롬프트 템플릿 설정
        self._setup_prompt_template()
        
        logger.info(f"✅ QWEN Wrapper initialized: {self.model_name}")
    
    def _load_model(self):
        """QWEN 모델과 토크나이저 로드"""
        try:
            logger.info(f"📥 Loading QWEN VL model: {self.model_name}")
            
            # Distributed training 환경 변수 설정 (자동 distributed 방지)
            import os
            if 'RANK' not in os.environ:
                os.environ['RANK'] = '0'
            if 'WORLD_SIZE' not in os.environ:
                os.environ['WORLD_SIZE'] = '1'
            if 'LOCAL_RANK' not in os.environ:
                os.environ['LOCAL_RANK'] = '0'
            if 'MASTER_ADDR' not in os.environ:
                os.environ['MASTER_ADDR'] = 'localhost'
            if 'MASTER_PORT' not in os.environ:
                os.environ['MASTER_PORT'] = '12355'
            
            # VL 모델을 위한 임포트
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            # 프로세서 로드 (VL 모델은 processor 사용)
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.tokenizer = self.processor.tokenizer
            
            # VL 모델 로드 (distributed 모드 방지)
            model_kwargs = {
                'torch_dtype': torch.float16 if self.device.type in ['cuda', 'mps'] else torch.float32,
                'trust_remote_code': True,
                'low_cpu_mem_usage': True
            }
            
            # device_map을 특정 GPU로 고정 (auto 사용 안함)
            if self.device.type == "cuda":
                model_kwargs['device_map'] = {
                    '': self.device  # 전체 모델을 지정된 GPU로
                }
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # CPU로 이동 (필요한 경우)
            if self.device.type != "cuda":
                self.model = self.model.to(self.device)
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 생성 설정
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _setup_prompt_template(self):
        """프롬프트 템플릿 설정 - user_prompt + placeholder 방식"""
        # Placeholder 템플릿 (user_prompt 뒤에 추가됨)
        self.placeholder_template = ", high quality, detailed, professional photography, cinematic lighting, artistic composition, 4k resolution"
        
        # 생성 지시를 위한 시스템 프롬프트
        self.system_prompt = """You are an expert at enhancing image generation prompts. 
Given a user prompt with a placeholder, complete the prompt by adding artistic details, style descriptions, and technical specifications.

Guidelines:
- Keep the original user prompt unchanged
- Add detailed descriptions after the placeholder
- Include artistic style, lighting, composition details
- Make it suitable for high-quality image generation
- Be concise but descriptive"""

        # 사용자 입력 템플릿 (user_prompt + placeholder)
        self.user_template = """{user_prompt_with_placeholder}

Complete this prompt with detailed enhancements: """
    
    def enhance_prompt(self, user_prompt: str) -> Dict[str, str]:
        """
        User prompt + placeholder를 향상된 prompt로 변환
        
        Args:
            user_prompt (str): 원본 사용자 프롬프트
            
        Returns:
            Dict[str, str]: 원본, placeholder 추가, 최종 향상된 프롬프트를 포함한 결과
        """
        try:
            # Step 1: user_prompt + placeholder 생성
            user_prompt_with_placeholder = user_prompt + self.placeholder_template
            
            # Step 2: VLM에 입력할 메시지 구성
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_template.format(user_prompt_with_placeholder=user_prompt_with_placeholder)}
            ]
            
            # 템플릿 적용
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 토크나이징
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            # 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # 디코딩
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # 후처리
            enhanced_prompt = self._post_process_output(generated_text)
            
            result = {
                'original_prompt': user_prompt,
                'prompt_with_placeholder': user_prompt_with_placeholder,
                'enhanced_prompt': enhanced_prompt,
                'raw_output': generated_text
            }
            
            logger.info(f"✨ Enhanced prompt: '{user_prompt}' + placeholder -> '{enhanced_prompt[:50]}...'")
            return result
            
        except Exception as e:
            logger.error(f"❌ Prompt enhancement failed: {e}")
            # 실패 시 placeholder만 추가된 버전 반환
            user_prompt_with_placeholder = user_prompt + self.placeholder_template
            return {
                'original_prompt': user_prompt,
                'prompt_with_placeholder': user_prompt_with_placeholder,
                'enhanced_prompt': user_prompt_with_placeholder,
                'raw_output': f"Error: {e}"
            }
    
    def _post_process_output(self, raw_output: str) -> str:
        """생성된 출력 후처리"""
        # 불필요한 텍스트 제거
        enhanced = raw_output.strip()
        
        # "Enhanced prompt:" 등의 레이블 제거
        enhanced = re.sub(r'^(Enhanced prompt:|Prompt:|Result:)\s*', '', enhanced, flags=re.IGNORECASE)
        enhanced = enhanced.strip()
        
        # 따옴표 제거
        enhanced = enhanced.strip('"\'')
        
        # 빈 결과 처리
        if not enhanced:
            return "high quality, detailed"
        
        return enhanced
    
    def enhance_prompts_batch(self, user_prompts: List[str]) -> List[Dict[str, str]]:
        """배치로 여러 프롬프트 향상"""
        results = []
        for prompt in user_prompts:
            result = self.enhance_prompt(prompt)
            results.append(result)
        return results
    
    def save_model(self, save_path: str):
        """모델 저장"""
        try:
            import os
            os.makedirs(save_path, exist_ok=True)
            
            # 모델과 토크나이저 저장
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            logger.info(f"💾 Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'max_new_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'vocab_size': len(self.tokenizer)
        } 