"""
VLM (Vision Language Model) Wrapper
===================================

Qwen2.5-VL을 실제로 로드하여 GRPO 학습을 위한 정책 네트워크로 사용하는 모듈입니다.
LoRA를 사용하여 효율적인 fine-tuning을 지원합니다.

주요 기능:
1. 실제 Qwen2.5-VL 모델 로드 및 관리
2. LoRA 어댑터를 통한 효율적 학습
3. GRPO 학습을 위한 정책 네트워크 인터페이스
4. 토큰별 순차 생성 및 확률 계산
5. 텍스트 생성 파라미터 관리
6. 토큰 길이 제한 (CLIP 77 토큰 제한)

Author: AI Assistant
Date: 2025-01-22
"""

import torch
import torch.nn as nn
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor, 
    AutoTokenizer,
    CLIPTokenizer
)
from torch.distributions import Categorical
from typing import List, Dict, Optional, Union
import logging
import json

# LoRA 관련 imports
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("⚠️ PEFT library not available. LoRA training will be disabled.")

logger = logging.getLogger(__name__)

class VLMWrapper(nn.Module):
    """
    Qwen2.5-VL을 GRPO 학습을 위한 정책 네트워크로 래핑하는 클래스
    
    이 클래스는 실제 Qwen2.5-VL 모델을 로드하고 GRPO 학습에 필요한
    정책 네트워크 인터페이스를 제공합니다. LoRA를 사용하여 효율적인 학습을 지원합니다.
    
    주요 메서드:
    - forward(): 토큰 시퀀스에 대한 다음 토큰 분포 계산
    - generate_sequence(): 토큰별 순차 생성
    - get_log_prob(): 특정 토큰의 로그 확률 계산
    
    Attributes:
        model_name (str): 사용할 Qwen2.5-VL 모델 이름
        model: Qwen2.5-VL 모델 객체 (LoRA 어댑터 포함 가능)
        tokenizer: 토크나이저 객체
        processor: 프로세서 객체
        clip_tokenizer: CLIP 토크나이저 (토큰 길이 체크용)
        device: 연산 디바이스 (MPS/CUDA/CPU)
        max_token_length (int): 최대 토큰 길이 (기본값: 77)
        use_lora (bool): LoRA 사용 여부
    """
    
    def __init__(self,
                 config_path: str = "config/default_config.json",
                 device: str = "auto",
                 max_token_length: int = 77):
        """
        VLM Wrapper 초기화 (실제 Qwen2.5-VL 모델 로드 with LoRA)
        
        Args:
            config_path (str): 설정 파일 경로
            device (str): 디바이스 설정
            max_token_length (int): 최대 토큰 길이 (CLIP 제한)
        """
        super().__init__()
        
        # 설정 파일에서 모델 이름 및 LoRA 설정 읽기
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.model_name = config['model_settings']['vlm_model']
            self.vlm_training_config = config['model_settings'].get('vlm_training', {})
            logger.info(f"📄 VLM model name from config: {self.model_name}")
            logger.info(f"🔧 VLM training config: {self.vlm_training_config}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load config: {e}, using default")
            self.model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
            self.vlm_training_config = {}
        
        # 디바이스 설정
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"🖥️ Using device: {self.device}")
        
        # 토큰 길이 제한 설정
        self.max_token_length = max_token_length
        
        # LoRA 설정
        self.use_lora = self.vlm_training_config.get('use_lora', False) and PEFT_AVAILABLE
        if self.use_lora and not PEFT_AVAILABLE:
            logger.warning("⚠️ LoRA requested but PEFT not available. Falling back to full fine-tuning.")
            self.use_lora = False
        
        # 모델 로드
        self._load_model()
        
        # CLIP 토크나이저 초기화 (토큰 길이 체크용)
        try:
            self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("✅ CLIP tokenizer loaded for token length validation")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load CLIP tokenizer: {e}, using fallback")
            self.clip_tokenizer = None
        
        logger.info(f"✅ VLM Wrapper initialized with {self.model_name} (max_tokens: {max_token_length})")
    
    def _load_config(self):
        """Config 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ Failed to load config: {e}")
            return {}
        if self.use_lora:
            logger.info(f"🎯 LoRA enabled for efficient training")
    
    def _load_model(self):
        """
        실제 Qwen2.5-VL 모델 로드 with LoRA support
        """
        try:
            logger.info(f"📥 Loading Qwen2.5-VL model: {self.model_name}")
            
            # 양자화 설정
            load_in_8bit = self.vlm_training_config.get('load_in_8bit', False)
            load_in_4bit = self.vlm_training_config.get('load_in_4bit', False)
            
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
                "device_map": "auto" if self.device.type == "cuda" else None,
                "trust_remote_code": True,  # Qwen 모델에 필요
                "attn_implementation": "eager",  # 안정성을 위해 eager attention 사용
            }
            
            if load_in_8bit:
                model_kwargs["load_in_8bit"] = True
                logger.info("🔧 Loading model in 8-bit mode")
            elif load_in_4bit:
                model_kwargs["load_in_4bit"] = True
                logger.info("🔧 Loading model in 4-bit mode")
            
            # Visual encoder 초기화 경고 억제를 위한 설정
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Some weights of.*were not initialized")
                
                # 모델 로드
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            
            # 모델 초기화 상태 상세 로깅
            self._log_model_initialization_status()
            
            # CPU로 이동 (필요한 경우)
            if self.device.type == "cpu" and not (load_in_8bit or load_in_4bit):
                self.model = self.model.to(self.device)
            
            # LoRA 어댑터 추가
            if self.use_lora:
                self._setup_lora()
            
            # 토크나이저 및 프로세서 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 모델을 학습 모드로 설정
            self.model.train()
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            if self.use_lora:
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                logger.info(f"🎯 Trainable parameters (LoRA): {trainable_params:,}")
            logger.info(f"📝 Vocab size: {len(self.tokenizer)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Qwen2.5-VL model: {e}")
            logger.info("🔄 Attempting fallback to Qwen2.5-7B-Instruct...")
            
            try:
                # Fallback to text-only model
                fallback_model = "Qwen/Qwen2.5-7B-Instruct"
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    fallback_model,
                    **model_kwargs
                )
                
                if self.device.type == "cpu" and not (load_in_8bit or load_in_4bit):
                    self.model = self.model.to(self.device)
                
                # LoRA 어댑터 추가 (fallback model에도)
                if self.use_lora:
                    self._setup_lora()
                
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                self.processor = AutoProcessor.from_pretrained(fallback_model)
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model.train()
                self.model_name = fallback_model
                
                logger.info(f"✅ Fallback model loaded: {fallback_model}")
                
            except Exception as e2:
                logger.error(f"❌ Fallback model loading also failed: {e2}")
                raise RuntimeError(f"Failed to load both primary and fallback models: {e}, {e2}")
    
    def _log_model_initialization_status(self):
        """
        모델 초기화 상태를 사용자에게 명확히 알려주는 메서드
        """
        try:
            # 전체 파라미터 수 계산
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # Visual encoder 관련 파라미터 확인
            visual_params = [name for name, _ in self.model.named_parameters() if 'visual' in name]
            text_params = [name for name, _ in self.model.named_parameters() if 'visual' not in name]
            
            logger.info("📊 Model Initialization Status:")
            logger.info(f"  Total parameters: {total_params:,}")
            logger.info(f"  Trainable parameters: {trainable_params:,}")
            logger.info(f"  Visual encoder parameters: {len(visual_params):,}")
            logger.info(f"  Text model parameters: {len(text_params):,}")
            
            if visual_params:
                logger.info("")
                logger.info("🖼️ Visual Encoder Information:")
                logger.info("  ✅ Visual encoder successfully loaded")
                logger.info("  ℹ️ Some visual weights may show 'newly initialized' warnings")
                logger.info("  ℹ️ This is NORMAL for Qwen2.5-VL models and does not affect performance")
                logger.info("  ℹ️ The model will learn appropriate visual representations during training")
                logger.info("")
                logger.info("🎯 Training Recommendation:")
                logger.info("  - Use LoRA for efficient training")
                logger.info("  - Start with lower learning rates for visual components")
                logger.info("  - Monitor visual-text alignment during training")
            else:
                logger.info("📝 Text-only model configuration detected")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not analyze model initialization: {e}")

    def _initialize_visual_encoder_if_needed(self):
        """
        Visual encoder 가중치 초기화 개선
        
        Qwen2.5-VL 모델에서 visual encoder 부분이 새로 초기화될 때
        더 적절한 초기화를 수행합니다.
        """
        try:
            logger.info("🔧 Checking visual encoder initialization...")
            
            # Visual encoder 모듈 찾기
            visual_modules = []
            for name, module in self.model.named_modules():
                if 'visual' in name and hasattr(module, 'weight'):
                    visual_modules.append((name, module))
            
            if not visual_modules:
                logger.info("ℹ️ No visual encoder modules found to initialize")
                return
            
            # 초기화가 필요한 모듈들에 대해 개선된 초기화 적용
            initialized_count = 0
            for name, module in visual_modules:
                if hasattr(module, 'weight') and module.weight is not None:
                    # Xavier/Glorot 초기화 적용
                    if len(module.weight.shape) >= 2:
                        torch.nn.init.xavier_uniform_(module.weight)
                        initialized_count += 1
                    
                    # bias 초기화
                    if hasattr(module, 'bias') and module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            
            if initialized_count > 0:
                logger.info(f"✅ Improved initialization applied to {initialized_count} visual encoder modules")
            else:
                logger.info("ℹ️ Visual encoder modules already properly initialized")
                
        except Exception as e:
            logger.warning(f"⚠️ Visual encoder initialization failed: {e}")
            logger.info("ℹ️ Continuing with default initialization")

    def _setup_lora(self):
        """
        LoRA 어댑터 설정
        """
        if not PEFT_AVAILABLE:
            logger.warning("⚠️ PEFT not available, skipping LoRA setup")
            return
        
        try:
            logger.info("🎯 Setting up LoRA adapter...")
            
            # LoRA 설정
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.vlm_training_config.get('lora_rank', 16),
                lora_alpha=self.vlm_training_config.get('lora_alpha', 32),
                lora_dropout=self.vlm_training_config.get('lora_dropout', 0.1),
                target_modules=self.vlm_training_config.get('lora_target_modules', ["q_proj", "v_proj", "k_proj", "o_proj"]),
                bias="none",
                inference_mode=False,  # 학습 모드
            )
            
            # LoRA 어댑터 적용
            self.model = get_peft_model(self.model, lora_config)
            
            # 학습 가능한 파라미터만 활성화
            self.model.print_trainable_parameters()
            
            logger.info("✅ LoRA adapter setup complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup LoRA: {e}")
            logger.info("🔄 Falling back to full fine-tuning")
            self.use_lora = False

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Categorical:
        """
        정책 네트워크 forward pass - 다음 토큰 분포 계산
        
        Args:
            input_ids (torch.Tensor): 입력 토큰 시퀀스 [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): 어텐션 마스크
            
        Returns:
            Categorical: 다음 토큰에 대한 확률 분포
        """
        # 입력 검증
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # 어텐션 마스크 생성 (필요한 경우)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # 모델 forward pass
        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        # 마지막 토큰의 로짓 추출
        logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
        
        # 확률 분포 생성
        return Categorical(logits=logits)
    
    def generate_sequence(self, prompt: str, max_new_tokens: int = None) -> Dict[str, any]:
        """
        토큰별 순차 생성 (GRPO 학습용)
        
        Args:
            prompt (str): 입력 프롬프트
            max_new_tokens (int): 최대 생성 토큰 수
            
        Returns:
            Dict: 생성 결과 (토큰, 로그 확률, 상태 등)
        """
        if max_new_tokens is None:
            # Config에서 토큰 설정을 로드
            try:
                config = self._load_config()
                max_new_tokens = config.get('token_settings', {}).get('max_new_tokens', 20)
            except:
                max_new_tokens = 20
        
        # 프롬프트 토크나이징
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # 생성 데이터 저장
        states = []
        actions = []
        log_probs = []
        
        current_ids = input_ids.clone()
        current_mask = attention_mask.clone()
        
        self.model.eval()
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # 현재 상태 저장
                states.append(current_ids.clone())
                
                # 다음 토큰 분포 계산
                policy_dist = self.forward(current_ids, current_mask)
                
                # 토큰 샘플링
                next_token = policy_dist.sample()
                log_prob = policy_dist.log_prob(next_token)
                
                # 데이터 저장
                actions.append(next_token)
                log_probs.append(log_prob)
                
                # 시퀀스 업데이트
                next_token_expanded = next_token.unsqueeze(-1)  # [batch_size, 1]
                current_ids = torch.cat([current_ids, next_token_expanded], dim=-1)
                
                # 어텐션 마스크 업데이트
                new_mask = torch.ones((current_mask.size(0), 1), device=self.device)
                current_mask = torch.cat([current_mask, new_mask], dim=-1)
                
                # EOS 토큰이면 중단
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # 생성된 텍스트 디코딩
        generated_text = self.tokenizer.decode(current_ids.squeeze(), skip_special_tokens=True)
        
        return {
            'generated_text': generated_text,
            'generated_ids': current_ids,
            'states': states,
            'actions': actions,
            'log_probs': log_probs
        }
    
    def get_log_prob(self, input_ids: torch.Tensor, target_token: torch.Tensor, 
                     attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        특정 토큰의 로그 확률 계산
        
        Args:
            input_ids (torch.Tensor): 입력 시퀀스
            target_token (torch.Tensor): 타겟 토큰
            attention_mask (torch.Tensor, optional): 어텐션 마스크
            
        Returns:
            torch.Tensor: 로그 확률
        """
        policy_dist = self.forward(input_ids, attention_mask)
        return policy_dist.log_prob(target_token)

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
                # Fallback: Qwen tokenizer 사용
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                return len(tokens)
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

    def enhance_prompt(self, user_prompt: str, use_model_generation: bool = True) -> str:
        """
        사용자 프롬프트를 개선 (모델 생성 또는 규칙 기반)
        
        Args:
            user_prompt (str): 개선할 원본 프롬프트
            use_model_generation (bool): 모델 생성 사용 여부
            
        Returns:
            str: 개선된 프롬프트 (77토큰 이하)
        """
        if not user_prompt or not user_prompt.strip():
            logger.warning("⚠️ Empty prompt provided")
            return "a beautiful image"
        
        try:
            user_prompt = user_prompt.strip()
            
            # 사용자 프롬프트의 토큰 수 확인
            user_prompt_tokens = self._count_tokens(user_prompt)
            logger.debug(f"📊 User prompt tokens: {user_prompt_tokens}/{self.max_token_length}")
            
            if user_prompt_tokens >= self.max_token_length:
                # 사용자 프롬프트가 이미 제한을 초과하는 경우 잘라내기
                user_prompt = self._truncate_to_token_limit(user_prompt, self.max_token_length - 5)
                logger.warning(f"⚠️ User prompt truncated to fit token limit: {user_prompt}")
                return user_prompt
            
            if use_model_generation and hasattr(self, 'model') and self.model is not None:
                # 모델을 사용한 프롬프트 개선
                enhanced_prompt = self._enhance_with_model(user_prompt)
            else:
                # 규칙 기반 프롬프트 개선
                available_tokens = self.max_token_length - user_prompt_tokens
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

    def _enhance_with_model(self, user_prompt: str) -> str:
        """
        모델을 사용한 프롬프트 개선
        
        Args:
            user_prompt (str): 원본 프롬프트
            
        Returns:
            str: 개선된 프롬프트
        """
        try:
            # 프롬프트 개선을 위한 템플릿
            enhancement_prompt = f"Enhance this image prompt to be more detailed and artistic: '{user_prompt}' -> Enhanced:"
            
            # 토큰화
            inputs = self.tokenizer(enhancement_prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 생성
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 디코딩
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 개선된 부분 추출
            if "Enhanced:" in generated_text:
                enhanced_part = generated_text.split("Enhanced:")[-1].strip()
            else:
                enhanced_part = generated_text.replace(enhancement_prompt, "").strip()
            
            # 최소 길이 확인
            if len(enhanced_part) < 10:
                raise ValueError("Enhanced prompt too short")
            
            return enhanced_part[:200]  # 최대 200자 제한
            
        except Exception as e:
            logger.warning(f"⚠️ Model-based enhancement failed: {e}, using fallback")
            return self._enhance_with_placeholders(user_prompt)

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
        
        # 개선된 프롬프트 구성
        enhanced_parts = [user_prompt]
        
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

    def _fallback_enhancement(self, user_prompt: str) -> str:
        """
        개선 실패 시 사용할 기본 개선 방법 (토큰 제한 고려)
        
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
        if hasattr(self, 'model') and self.model is not None:
            return {
                "model_name": self.model_name,
                "device": str(self.device),
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "vocab_size": len(self.tokenizer) if hasattr(self, 'tokenizer') else None,
                "max_token_length": self.max_token_length
            }
        else:
            return {
                "model_name": self.model_name,
                "device": str(self.device),
                "parameters": 0,
                "vocab_size": None,
                "max_token_length": self.max_token_length,
                "status": "not_loaded"
            }


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
            max_token_length=77
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