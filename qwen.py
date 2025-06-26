import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import copy
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, GenerationConfig
from peft import LoraConfig, get_peft_model, TaskType
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class QWENGRPOConfig:
    """QWEN GRPO 통합 설정 (CartPole GRPO 호환 + EasyR1 안정성)"""
    learning_rate: float = 2e-4  # LoRA는 더 높은 학습률 사용 가능
    batch_size: int = 2  # 메모리 절약을 위해 배치 크기 감소 (8 → 2)
    num_rollouts: int = 2  # 메모리 절약을 위해 롤아웃 수 감소 (6 → 2)
    max_prompt_length: int = 77
    max_new_tokens: int = 20
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 100
    kl_coef: float = 0.02
    clip_ratio: float = 0.1
    entropy_coef: float = 0.02
    save_images: bool = True
    log_dir: str = "grpo_results"
    
    # CartPole GRPO 호환 추가 설정
    gamma: float = 0.995  # 할인 팩터 (VLM은 CartPole보다 높게)
    grpo_epochs: int = 10  # 다중 에포크 업데이트
    update_ref_model_freq: int = 1  # Reference 모델 업데이트 빈도
    epsilon_std: float = 1e-8  # 정규화 안정성
    
    # EasyR1 스타일 수치적 안정성 설정
    use_adaptive_grad_clip: bool = True  # 적응적 그래디언트 클리핑
    grad_clip_ema_beta: float = 0.99  # 그래디언트 norm EMA 계수
    grad_clip_coef: float = 1.5  # 적응적 클리핑 계수
    use_grad_centralization: bool = True  # 그래디언트 중앙화
    use_grad_normalization: bool = True  # 그래디언트 정규화
    grad_norm_alpha: float = 0.5  # 정규화 강도
    use_stochastic_rounding: bool = True  # 확률적 반올림 (시뮬레이션)
    logits_clip_range: float = 20.0  # logits 클리핑 범위 (더 보수적으로)
    stable_log_prob_min: float = -50.0  # 안전한 로그 확률 최소값

class QWENModel:

    def __init__(self, model_name = "Qwen/Qwen2-VL-7B-Instruct", device = "cuda", temperature = 0.7, grpo_config: QWENGRPOConfig = None):
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.grpo_config = grpo_config or QWENGRPOConfig()  # Accelerate 메인 프로세스 여부

        self._load_model()
        self._setup_prompt_template()
        
        # GRPO 관련 초기화
        if grpo_config:
            self._setup_grpo_components()
        
        logger.info(f"Qwen init : {self.model_name}")

    def _load_model(self):
        
        self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
        self.tokenizer = self.processor.tokenizer
  
        # Accelerate 호환을 위한 모델 로딩 설정
        if self.device in ["cpu", "accelerate"]:
            # Accelerate가 관리할 경우 device_map 없이 로딩
            model_kwargs = {
                'torch_dtype': torch.float16,
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,
                # device_map을 제거하여 Accelerate가 분산 관리하도록 함
            }
        else:
            # 기본 GPU 사용 시 (단일 GPU 모드) - 메모리 최적화
            model_kwargs = {
                'torch_dtype': torch.float16,
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,
                'use_cache': False,  # 캐시 비활성화로 메모리 절약
                # device_map 제거하여 단일 GPU 사용
            }

        logger.info("🔧 QWEN 7B 모델 로딩 중... (단일 GPU 모드)")
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
        )
        
        # 모델을 지정된 GPU로 이동 (Accelerate 모드가 아닐 때만)
        if self.device != "accelerate":
            self.model = self.model.to(self.device)
            logger.info(f"✅ 모델을 {self.device}로 이동")
        
        # LoRA 설정 및 적용 - 고성능 설정 (메모리 여유분 활용)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,  # LoRA rank 대폭 감소 (32 → 8) - 메모리 절약
            lora_alpha=16,  # LoRA scaling parameter 감소 (64 → 16)
            lora_dropout=0.1,  # 드롭아웃 증가
            target_modules=[
                # Attention 모듈들만 (MLP 제거로 메모리 절약)
                "q_proj", "v_proj", "k_proj", "o_proj",
                # Vision 관련 모듈들 추가
                "visual_proj", "lm_head"
            ],
            bias="lora_only",  # bias도 LoRA로 학습
            inference_mode=False,
            modules_to_save=["embed_tokens"],  # 임베딩도 학습
        )
        
        # LoRA 어댑터 적용
        self.model = get_peft_model(self.model, lora_config)
        logger.info("✅ LoRA 어댑터 적용 완료")
        
        # LoRA 파라미터 정보 출력 (상세 분석)
        try:
            # 전체 파라미터 수 계산
            total_params = sum(p.numel() for p in self.model.parameters())
            
            # 학습 가능한 파라미터 수 계산
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # 각 LoRA 레이어별 파라미터 수 분석
            lora_details = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad and ('lora' in name.lower() or 'adapter' in name.lower()):
                    module_name = name.split('.')[0] if '.' in name else name
                    if module_name not in lora_details:
                        lora_details[module_name] = 0
                    lora_details[module_name] += param.numel()
            
            trainable_percentage = 100 * trainable_params / total_params if total_params > 0 else 0
            
            logger.info(f"📊 LoRA 파라미터 상세 정보:")
            logger.info(f"  - 전체 모델 파라미터: {total_params:,}")
            logger.info(f"  - 학습 가능한 파라미터: {trainable_params:,}")
            logger.info(f"  - 학습 비율: {trainable_percentage:.4f}%")
            logger.info(f"  - LoRA 설정:")
            logger.info(f"    * Rank (r): {lora_config.r}")
            logger.info(f"    * Alpha: {lora_config.lora_alpha}")
            logger.info(f"    * Dropout: {lora_config.lora_dropout}")
            logger.info(f"    * 타겟 모듈: {lora_config.target_modules}")
            
            if lora_details:
                logger.info(f"  - LoRA 레이어별 파라미터:")
                for module, count in lora_details.items():
                    logger.info(f"    * {module}: {count:,} 파라미터")
                    
        except Exception as e:
            logger.warning(f"⚠️ LoRA 파라미터 정보 출력 실패: {e}")
            logger.info("✅ LoRA 어댑터는 정상적으로 적용됨")
        
        # 메모리 최적화 설정
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing 활성화")

        if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 생성 설정 - 극한 메모리 최적화
        self.generation_config = GenerationConfig(
            max_new_tokens=20,  # 토큰 수 더 제한 (30 → 20)
            temperature=self.temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=False,  # 메모리 절약을 위해 캐시 비활성화 유지
            output_hidden_states=False,  # hidden states 출력 비활성화
            output_attentions=False,  # attention weights 출력 비활성화
        )
        
        logger.info("✅ QWEN 모델 로드 완료 (메모리 최적화 적용)")

    def _setup_prompt_template(self):
         
        # 생성 지시를 위한 시스템 프롬프트
        self.system_prompt = """You are an expert at enhancing image generation prompts. 
        Given a simple user prompt, expand it into a detailed, and high-quality prompt for image generation.

        Guidelines:
        - Keep the original concept unchanged
        - Only using English words and sentences
        - Do not use any other languages
        - Do not use any special characters
        - Add artistic style, mood, and atmosphere
        - Include technical specifications (lighting, composition, resolution)
        - Add creative details that make the image more realistic
        - Make each enhancement unique and varied
        - Be descriptive but concise (aim for 20-40 additional words) """

        # 사용자 입력 템플릿
        self.user_template = """Original prompt: {user_prompt}

        Enhanced version:"""

    def _setup_grpo_components(self):
        """GRPO 관련 컴포넌트 설정 - 메모리 최적화 버전"""
        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = len(self.tokenizer.get_vocab())
        
        logger.info("🔧 GRPO 컴포넌트 초기화 중... (메모리 최적화)")
        
        # Reference 모델은 항상 활성화 (KL penalty 필요)
        logger.info("🎯 Reference 모델 활성화 (KL penalty 계산용)")
        
        # Reference 모델을 CLIP GPU로 이동 (GPU 1) - 단일 GPU 모드
        logger.info("🔧 Reference 모델 생성 중... (CLIP GPU로 이동)")
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        
        # Reference 모델을 GPU 1 (CLIP GPU)로 이동
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.ref_model = self.ref_model.to("cuda:1")
            logger.info("✅ Reference 모델을 GPU 1 (CLIP GPU)로 이동")
        else:
            logger.info("✅ Reference 모델 생성 완료 (현재 디바이스)")
        
        # 옵티마이저 (LoRA 파라미터만 학습)
        # LoRA 파라미터만 학습하도록 필터링
        trainable_params = []
        total_trainable_params = 0
        
        logger.info("🔍 LoRA 학습 가능한 파라미터 분석:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                param_count = param.numel()
                total_trainable_params += param_count
                logger.info(f"  📌 {name}: {param_count:,} 파라미터 (shape: {param.shape})")
        
        logger.info(f"📊 LoRA 학습 파라미터 총계:")
        logger.info(f"  - 학습 가능한 레이어 수: {len(trainable_params)}")
        logger.info(f"  - 총 학습 파라미터 수: {total_trainable_params:,}")
        
        self.grpo_optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=self.grpo_config.learning_rate,
            weight_decay=0.01
        )
        
        # EasyR1 스타일 수치적 안정성 변수 초기화
        self.grad_norm_ema = 0.0  # 그래디언트 norm의 지수 이동 평균
        self.training_step = 0  # 트레이닝 스텝 카운터
        
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 GPU 메모리 캐시 정리 완료")
        
        logger.info("✅ GRPO 컴포넌트 초기화 완료 (메모리 최적화 + EasyR1 안정성)")
        logger.info(f"📊 QWEN 모델 직접 학습 방식으로 설정")
        logger.info(f"🔧 EasyR1 안정성 기법:")
        logger.info(f"  - 적응적 그래디언트 클리핑: {self.grpo_config.use_adaptive_grad_clip}")
        logger.info(f"  - 그래디언트 중앙화: {self.grpo_config.use_grad_centralization}")
        logger.info(f"  - 확률적 반올림: {self.grpo_config.use_stochastic_rounding}")
        logger.info(f"  - Logits 클리핑 범위: ±{self.grpo_config.logits_clip_range}")
        logger.info(f"  - 안전한 로그 확률 최소값: {self.grpo_config.stable_log_prob_min}")

    def _init_grpo_weights(self):
        """GRPO 가중치 초기화 (QWEN 직접 학습 방식에서는 불필요)"""
        pass
    
    def _get_model_for_generation(self):
        """Accelerate/DDP 래핑된 모델에서 원본 모델 가져오기"""
        # DistributedDataParallel로 래핑된 경우 .module로 접근
        if hasattr(self.model, 'module'):
            return self.model.module
        # 일반 모델인 경우 그대로 반환
        return self.model

    def enhance_prompt(self, user_prompt):
        """기본 프롬프트 향상 (GRPO 없이)"""
        # VLM에 입력할 메시지 구성
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(user_prompt=user_prompt)}
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
        )
        
        # Accelerate 환경이 아닐 때만 명시적으로 디바이스 이동
        if self.device != "accelerate":
            inputs = inputs.to(self.device)
        
        # 생성
        with torch.no_grad():
            model_for_gen = self._get_model_for_generation()
            outputs = model_for_gen.generate(
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
            'enhanced_prompt': enhanced_prompt,
            'raw_output': generated_text
        }
        
        logger.info(f"Enhanced prompt: '{user_prompt}' -> '{enhanced_prompt[:50]}...'")
        return result

    def generate_enhancement_candidates(self, user_prompt: str, num_candidates: int = None, 
                                        use_semantic_filtering: bool = True, 
                                        semantic_threshold: float = 0.7) -> List[str]:
        """여러 개의 향상된 프롬프트 후보 생성 (semantic filtering 포함)"""
        if num_candidates is None:
            num_candidates = self.grpo_config.num_enhancement_candidates
        
        candidates = []
        
        # VLM에 입력할 메시지 구성
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(user_prompt=user_prompt)}
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
        )
        
        # Accelerate 환경이 아닐 때만 명시적으로 디바이스 이동
        if self.device != "accelerate":
            inputs = inputs.to(self.device)
        
        # Semantic filtering 제거 - GRPO가 직접 exploration과 exploitation 균형을 학습
        # 다양한 후보 생성을 위해 더 많은 시도
        total_generations = num_candidates * 2
        
        raw_candidates = []
        
        # 여러 후보 생성
        for i in range(total_generations):
            # 다양성을 위해 temperature와 top_p 조정
            temp_config = GenerationConfig(
                max_new_tokens=77,
                temperature=self.temperature + (i * 0.05),  # 더 세밀한 조정
                top_p=0.85 + (i % 3) * 0.05,  # 0.85, 0.9, 0.95 순환
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            with torch.no_grad():
                model_for_gen = self._get_model_for_generation()
                outputs = model_for_gen.generate(
                    **inputs,
                    generation_config=temp_config,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # 디코딩
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # 후처리
            enhanced_prompt = self._post_process_output(generated_text)
            
            # 중복 제거
            if enhanced_prompt not in raw_candidates and enhanced_prompt != user_prompt:
                raw_candidates.append(enhanced_prompt)
        
        # Semantic filtering 제거 - 다양한 후보들을 GRPO가 직접 평가하도록
        # 중복 제거하고 다양성 확보를 위해 셔플
        import random
        random.shuffle(raw_candidates)
        candidates = raw_candidates[:num_candidates]
        
        logger.info(f"🎯 다양한 후보 생성: {len(raw_candidates)}개 → {num_candidates}개 선택 (semantic filtering 없음)")
        
        # 후보가 부족하면 원본 프롬프트로 채움
        while len(candidates) < num_candidates:
            candidates.append(user_prompt)
            logger.warning(f"⚠️ 후보 부족으로 원본 프롬프트 추가: {len(candidates)}/{num_candidates}")
        
        return candidates[:num_candidates]

    def get_grpo_state_representation(self, user_prompt: str) -> torch.Tensor:
        """GRPO를 위한 상태 표현 생성 (QWEN 모델도 학습 가능)"""
        # VLM에 입력할 메시지 구성
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(user_prompt=user_prompt)}
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
        )
        
        # Accelerate 환경이 아닐 때만 명시적으로 디바이스 이동
        if self.device != "accelerate":
            inputs = inputs.to(self.device)
        
        # 히든 스테이트 추출 (QWEN 모델도 학습되도록 gradient 계산)
        outputs = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            output_hidden_states=True
        )
        
        if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        else:
            raise AttributeError("Cannot find hidden states in model output")
        
        # 마지막 토큰의 히든 스테이트 사용
        if inputs['attention_mask'] is not None:
            last_valid_indices = inputs['attention_mask'].sum(dim=1) - 1
            last_valid_indices = torch.clamp(last_valid_indices, min=0)
            state_repr = hidden_states[torch.arange(hidden_states.size(0)), last_valid_indices]
        else:
            state_repr = hidden_states[:, -1, :]
        
        return state_repr.squeeze(0)  # [hidden_size]

    def generate_grpo_enhanced_prompt(self, user_prompt: str) -> Tuple[str, torch.Tensor]:
        """GRPO를 통해 향상된 프롬프트 생성 (QWEN 직접 학습)"""
        # VLM에 입력할 메시지 구성
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(user_prompt=user_prompt)}
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
        )
        
        # Accelerate 환경이 아닐 때만 명시적으로 디바이스 이동
        if self.device != "accelerate":
            inputs = inputs.to(self.device)
        
        # QWEN 모델로 직접 생성 (gradient 계산)
        model_for_gen = self._get_model_for_generation()
        outputs = model_for_gen.generate(
            **inputs,
            generation_config=self.generation_config,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # 생성된 텍스트 디코딩
        generated_text = self.tokenizer.decode(
            outputs.sequences[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # 후처리
        enhanced_prompt = self._post_process_output(generated_text)
        
        # 로그 확률 계산 (생성된 토큰들의 총 로그 확률) - GRPO 정확한 계산
        if hasattr(outputs, 'scores') and outputs.scores:
            log_probs = []
            for i, score in enumerate(outputs.scores):
                try:
                    token_id = outputs.sequences[0][inputs['input_ids'].shape[1] + i]
                    
                    # 안전한 log_softmax 계산
                    # score에 inf, nan이 있는지 확인
                    if torch.isnan(score).any() or torch.isinf(score).any():
                        logger.warning(f"⚠️ Score에 nan/inf 발견, 클리핑 적용")
                        score = torch.clamp(score, min=-100, max=100)
                    
                    # log_softmax 계산
                    log_softmax_scores = F.log_softmax(score, dim=-1)
                    
                    # 결과 검증
                    if torch.isnan(log_softmax_scores).any() or torch.isinf(log_softmax_scores).any():
                        logger.warning(f"⚠️ Log softmax에 nan/inf 발견, 안전한 값으로 대체")
                        log_prob = torch.tensor(-10.0, device=score.device)  # 안전한 기본값
                    else:
                        log_prob = log_softmax_scores[0, token_id]
                        
                        # 추가 안전성 검사
                        if torch.isnan(log_prob) or torch.isinf(log_prob):
                            log_prob = torch.tensor(-10.0, device=score.device)
                    
                    log_probs.append(log_prob)
                    
                except Exception as e:
                    logger.warning(f"⚠️ 로그 확률 계산 오류: {e}, 기본값 사용")
                    if self.device == "accelerate":
                        log_probs.append(torch.tensor(-10.0))
                    else:
                        log_probs.append(torch.tensor(-10.0, device=self.device))
            
            if log_probs:
                # GRPO에서는 모든 토큰의 로그 확률 합을 사용 (평균이 아님)
                total_log_prob = torch.stack(log_probs).sum()
                
                # 최종 안전성 검사
                if torch.isnan(total_log_prob) or torch.isinf(total_log_prob):
                    logger.warning("⚠️ 총 로그 확률에 nan/inf 발견, 안전한 값으로 대체")
                    if self.device == "accelerate":
                        total_log_prob = torch.tensor(-10.0)
                    else:
                        total_log_prob = torch.tensor(-10.0, device=self.device)
                        
                logger.debug(f"🔍 Generation: {len(log_probs)} tokens, total_log_prob={total_log_prob:.4f}")
                avg_log_prob = total_log_prob  # 변수명 유지 (하위 호환성)
            else:
                # 로그 확률 계산 실패 시 기본값
                if self.device == "accelerate":
                    avg_log_prob = torch.tensor(-10.0)
                else:
                    avg_log_prob = torch.tensor(-10.0, device=self.device)
        else:
            # fallback: 안전한 기본 로그 확률
            if self.device == "accelerate":
                avg_log_prob = torch.tensor(-10.0)  # Accelerate가 디바이스 관리
            else:
                avg_log_prob = torch.tensor(-10.0, device=self.device)
        
        return enhanced_prompt, avg_log_prob

    def calculate_log_prob_for_grpo(self, user_prompt: str, enhanced_prompt: str) -> torch.Tensor:
        """현재 모델의 로그 확률 계산 (gradient 계산 필요) - GRPO 정확한 구현"""
        # VLM에 입력할 메시지 구성
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(user_prompt=user_prompt)}
        ]
        
        # 템플릿 적용
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 프롬프트만 토크나이징 (생성 시작점 확인용)
        prompt_inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        prompt_length = prompt_inputs['input_ids'].shape[1]
        
        # 전체 시퀀스 (프롬프트 + 생성된 텍스트)
        full_text = prompt + enhanced_prompt
        
        # 전체 시퀀스 토크나이징
        inputs = self.tokenizer(
            full_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        
        # Accelerate 환경이 아닐 때만 명시적으로 디바이스 이동
        if self.device != "accelerate":
            inputs = inputs.to(self.device)
        
        # 현재 모델로 로그 확률 계산 (gradient 계산)
        model_for_gen = self._get_model_for_generation()
        outputs = model_for_gen(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        
        # 생성된 부분의 로그 확률만 계산 - GRPO 정확한 방식
        try:
            # 생성된 토큰들 (프롬프트 이후 부분)
            generated_tokens = inputs['input_ids'][0, prompt_length:]
            
            if len(generated_tokens) == 0:
                logger.warning("⚠️ 생성된 토큰이 없음, 기본값 반환")
                return torch.tensor(-10.0, device=self.device, requires_grad=True)
            
            # 생성된 토큰에 대응하는 logits (shift by 1)
            # logits[i]는 token[i+1]을 예측하므로, prompt_length-1부터 시작
            generated_logits = outputs.logits[0, prompt_length-1:prompt_length-1+len(generated_tokens)]
            
            # EasyR1 스타일 안전성 검사 - 더 보수적인 클리핑
            if torch.isnan(generated_logits).any() or torch.isinf(generated_logits).any():
                logger.warning("⚠️ Generated logits에 nan/inf 발견, EasyR1 스타일 클리핑 적용")
                generated_logits = torch.clamp(generated_logits, 
                                               min=-self.grpo_config.logits_clip_range, 
                                               max=self.grpo_config.logits_clip_range)
            else:
                # 예방적 클리핑 (EasyR1 스타일)
                generated_logits = torch.clamp(generated_logits, 
                                               min=-self.grpo_config.logits_clip_range, 
                                               max=self.grpo_config.logits_clip_range)
            
            # 확률적 반올림 시뮬레이션 (EasyR1 스타일)
            if self.grpo_config.use_stochastic_rounding and self.training:
                # 작은 노이즈 추가로 stochastic rounding 효과 시뮬레이션
                noise = torch.randn_like(generated_logits) * 1e-6
                generated_logits = generated_logits + noise
            
            # 각 토큰에 대한 로그 확률 계산 (gradient 유지)
            log_probs = F.log_softmax(generated_logits, dim=-1)
            
            # log_softmax 결과 검증
            if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                logger.warning("⚠️ Log probabilities에 nan/inf 발견, 안전한 값으로 대체")
                return torch.tensor(-10.0, device=generated_logits.device, requires_grad=True)
            
            # 각 생성된 토큰의 로그 확률 추출
            token_log_probs = log_probs.gather(1, generated_tokens.unsqueeze(1)).squeeze(1)
            
            # GRPO에서는 모든 토큰의 로그 확률 합을 사용 (평균이 아님)
            total_log_prob = token_log_probs.sum()
            
            # 최종 결과 검증
            if torch.isnan(total_log_prob) or torch.isinf(total_log_prob):
                logger.warning("⚠️ 총 로그 확률에 nan/inf 발견, 안전한 값으로 대체")
                return torch.tensor(-10.0, device=generated_logits.device, requires_grad=True)
            
            logger.debug(f"🔍 Current model: {len(generated_tokens)} tokens, total_log_prob={total_log_prob:.4f}")
            return total_log_prob
            
        except Exception as e:
            logger.warning(f"⚠️ 로그 확률 계산 중 오류: {e}, 안전한 기본값 반환")
            if self.device == "accelerate":
                return torch.tensor(-10.0, requires_grad=True)
            else:
                return torch.tensor(-10.0, device=self.device, requires_grad=True)

    def get_ref_model_log_prob(self, user_prompt: str, enhanced_prompt: str) -> torch.Tensor:
        """참조 모델의 로그 확률 계산 (단일 GPU 모드)"""
        # Reference model이 없으면 더미 값 반환 (단일 GPU 모드에서는 항상 있어야 함)
        if self.ref_model is None:
            logger.warning("⚠️ Reference 모델이 없습니다!")
            return torch.tensor(0.0, device=self.device, dtype=torch.float16)
        
        # VLM에 입력할 메시지 구성
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(user_prompt=user_prompt)}
        ]
        
        # 템플릿 적용
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 전체 시퀀스 (프롬프트 + 생성된 텍스트)
        full_text = prompt + enhanced_prompt
        
        # Reference model의 디바이스 확인 (GPU 1에 있음)
        ref_device = next(self.ref_model.parameters()).device
        logger.info(f"🔍 Reference 모델 디바이스: {ref_device}")
        
        # 토크나이징 (Reference model 디바이스에 맞춤)
        inputs = self.tokenizer(
            full_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(ref_device)
        
        # 참조 모델로 로그 확률 계산
        with torch.no_grad():
            outputs = self.ref_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            # 생성된 부분의 로그 확률만 계산 - GRPO 정확한 방식
            # 프롬프트 길이 계산 (current model과 동일한 방식)
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
            prompt_length = prompt_inputs['input_ids'].shape[1]
            
            # 생성된 토큰들 (프롬프트 이후 부분)
            generated_tokens = inputs['input_ids'][0, prompt_length:]
            
            if len(generated_tokens) == 0:
                logger.warning("⚠️ Reference model: 생성된 토큰이 없음")
                return torch.tensor(-10.0, device=self.device)
            
            # 생성된 토큰에 대응하는 logits (shift by 1)
            generated_logits = outputs.logits[0, prompt_length-1:prompt_length-1+len(generated_tokens)]
            
            # EasyR1 스타일 안전한 로그 확률 계산
            if torch.isnan(generated_logits).any() or torch.isinf(generated_logits).any():
                logger.warning("⚠️ Reference model logits에 nan/inf 발견, EasyR1 스타일 클리핑 적용")
                generated_logits = torch.clamp(generated_logits, 
                                               min=-self.grpo_config.logits_clip_range, 
                                               max=self.grpo_config.logits_clip_range)
            else:
                # 예방적 클리핑 (EasyR1 스타일)
                generated_logits = torch.clamp(generated_logits, 
                                               min=-self.grpo_config.logits_clip_range, 
                                               max=self.grpo_config.logits_clip_range)
            
            log_probs = F.log_softmax(generated_logits, dim=-1)
            
            # log_softmax 결과 검증
            if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                logger.warning("⚠️ Reference log probabilities에 nan/inf 발견, 안전한 값으로 대체")
                return torch.tensor(-10.0, device=self.device)
            
            # 각 생성된 토큰의 로그 확률 추출
            token_log_probs = log_probs.gather(1, generated_tokens.unsqueeze(1)).squeeze(1)
            
            # GRPO에서는 모든 토큰의 로그 확률 합을 사용 (평균이 아님)
            total_log_prob = token_log_probs.sum()
            
            # 최종 결과 검증
            if torch.isnan(total_log_prob) or torch.isinf(total_log_prob):
                logger.warning("⚠️ Reference 총 로그 확률에 nan/inf 발견, 안전한 값으로 대체")
                return torch.tensor(-10.0, device=self.device)
            
            # 결과를 main model device로 이동
            result = total_log_prob.to(self.device)
            logger.debug(f"🔍 Reference model: {len(generated_tokens)} tokens, total_log_prob={result:.4f}")
            return result

    def calculate_discounted_returns(self, rewards: List[float], gamma: float = None) -> torch.Tensor:
        """할인된 리턴 계산 (CartPole GRPO 방식)"""
        if gamma is None:
            gamma = self.grpo_config.gamma
        
        returns = torch.zeros(len(rewards), device=self.device, dtype=torch.float32)
        discounted_return = 0.0
        
        # 역순으로 할인된 리턴 계산
        for t in reversed(range(len(rewards))):
            discounted_return = rewards[t] + gamma * discounted_return
            returns[t] = discounted_return
        
        return returns
    
    def calculate_normalized_advantages(self, all_returns: torch.Tensor) -> torch.Tensor:
        """전체 그룹에 대한 정규화 (CartPole GRPO 방식)"""
        if len(all_returns) <= 1:
            return all_returns
        
        mean_return = torch.mean(all_returns)
        std_return = torch.std(all_returns)
        
        # 정규화
        normalized_advantages = (all_returns - mean_return) / (std_return + self.grpo_config.epsilon_std)
        return normalized_advantages

    def update_grpo_policy(self, experiences: List[Dict]) -> Dict:
        """GRPO 정책 업데이트 (CartPole GRPO 호환)"""
        if not experiences:
            return {}
        
        # 경험 데이터 준비
        user_prompts = []
        enhanced_prompts = []
        old_log_probs = []
        rewards = []
        
        for exp in experiences:
            user_prompts.append(exp['user_prompt'])
            enhanced_prompts.append(exp['enhanced_prompt'])
            old_log_probs.append(exp['log_prob'])
            rewards.append(exp['reward'])
        
        # 텐서로 변환 (gradient 계산을 위해 float32 사용)
        old_log_probs = torch.stack(old_log_probs)
        
        # 할인된 리턴 계산 (CartPole GRPO 방식)
        discounted_returns = self.calculate_discounted_returns(rewards)
        
        # 정규화된 advantage 계산 (CartPole GRPO 방식)
        advantages = self.calculate_normalized_advantages(discounted_returns)
        
        logger.info(f"📊 할인된 리턴 통계:")
        logger.info(f"  원본 리워드: mean={sum(rewards)/len(rewards):.4f}, std={torch.tensor(rewards).std():.4f}")
        logger.info(f"  할인된 리턴: mean={discounted_returns.mean():.4f}, std={discounted_returns.std():.4f}")
        logger.info(f"  정규화된 advantage: mean={advantages.mean():.4f}, std={advantages.std():.4f}")
        
        baseline = sum(rewards) / len(rewards)  # 로깅용
        
        # 현재 모델과 참조 모델의 로그 확률 계산 (gradient 계산 필요)
        current_log_probs = []
        ref_log_probs = []
        
        logger.info("🔍 로그 확률 계산 중...")
        for i, (user_prompt, enhanced_prompt) in enumerate(zip(user_prompts, enhanced_prompts)):
            # 현재 모델의 로그 확률 계산 (gradient 필요)
            current_log_prob = self.calculate_log_prob_for_grpo(user_prompt, enhanced_prompt)
            current_log_probs.append(current_log_prob)
            
            # 참조 모델의 로그 확률 (gradient 불필요)
            ref_log_prob = self.get_ref_model_log_prob(user_prompt, enhanced_prompt)
            ref_log_probs.append(ref_log_prob)
            
            logger.info(f"  경험 {i+1}: current_log_prob={current_log_prob:.4f}, ref_log_prob={ref_log_prob:.4f}")
        
        current_log_probs = torch.stack(current_log_probs)
        ref_log_probs = torch.stack(ref_log_probs)
        
        logger.info(f"📊 로그 확률 통계:")
        logger.info(f"  Current log probs: mean={current_log_probs.mean():.4f}, std={current_log_probs.std():.4f}")
        logger.info(f"  Ref log probs: mean={ref_log_probs.mean():.4f}, std={ref_log_probs.std():.4f}")
        logger.info(f"  Old log probs: mean={old_log_probs.mean():.4f}, std={old_log_probs.std():.4f}")
        logger.info(f"  Advantages: mean={advantages.mean():.4f}, std={advantages.std():.4f}")
        
        # 중요도 비율 계산 - 안전한 계산
        log_ratio = current_log_probs - old_log_probs
        
        logger.info(f"🔍 중요도 비율 계산:")
        logger.info(f"  Log ratio: mean={log_ratio.mean():.6f}, std={log_ratio.std():.6f}")
        
        # 안전성 검사
        if torch.isnan(log_ratio).any() or torch.isinf(log_ratio).any():
            logger.warning("⚠️ Log ratio에 nan/inf 발견, 클리핑 적용")
            log_ratio = torch.clamp(log_ratio, min=-10, max=10)
        
        ratio = torch.exp(log_ratio)
        logger.info(f"  Ratio: mean={ratio.mean():.6f}, std={ratio.std():.6f}")
        
        # ratio 안전성 검사
        if torch.isnan(ratio).any() or torch.isinf(ratio).any():
            logger.warning("⚠️ Ratio에 nan/inf 발견, 안전한 값으로 대체")
            ratio = torch.clamp(ratio, min=0.1, max=10.0)
        
        # PPO 클립된 목적 함수
        clipped_ratio = torch.clamp(ratio, 1 - self.grpo_config.clip_ratio, 1 + self.grpo_config.clip_ratio)
        logger.info(f"  Clipped ratio: mean={clipped_ratio.mean():.6f}, std={clipped_ratio.std():.6f}")
        
        # 정책 손실 계산
        policy_obj1 = ratio * advantages
        policy_obj2 = clipped_ratio * advantages
        
        # 안전성 검사 - 0으로 설정하지 말고 작은 값으로 설정
        if torch.isnan(policy_obj1).any() or torch.isinf(policy_obj1).any():
            logger.warning("⚠️ Policy objective 1에 nan/inf 발견, 작은 값으로 대체")
            policy_obj1 = advantages * 0.01  # 작은 신호 유지
            
        if torch.isnan(policy_obj2).any() or torch.isinf(policy_obj2).any():
            logger.warning("⚠️ Policy objective 2에 nan/inf 발견, 작은 값으로 대체")
            policy_obj2 = advantages * 0.01  # 작은 신호 유지
        
        policy_loss = -torch.min(policy_obj1, policy_obj2).mean()
        
        logger.info(f"🔍 정책 손실 계산:")
        logger.info(f"  Policy objective 1 mean: {policy_obj1.mean():.6f}")
        logger.info(f"  Policy objective 2 mean: {policy_obj2.mean():.6f}")
        logger.info(f"  Policy loss: {policy_loss:.6f}")
        
        # KL 발산 페널티 (CartPole GRPO 정확한 방식)
        if self.ref_model is not None:
            # 정확한 KL divergence 추정기 (CartPole GRPO 방식)
            with torch.no_grad():
                log_ratio_ref_curr = ref_log_probs - current_log_probs.detach()
            
            # KL(ref || current) = exp(log_ratio) - log_ratio - 1
            kl_div_estimate = torch.exp(log_ratio_ref_curr) - log_ratio_ref_curr - 1
            kl_div = torch.relu(kl_div_estimate.mean())  # 음수 방지
            
            # KL divergence 안전성 검사
            if torch.isnan(kl_div) or torch.isinf(kl_div):
                logger.warning("⚠️ KL divergence에 nan/inf 발견, 안전한 값으로 대체")
                kl_div = torch.tensor(0.01, device=self.device)  # 작은 값으로 변경
            
            kl_penalty = self.grpo_config.kl_coef * kl_div
            logger.info(f"  KL divergence (정확한 추정): {kl_div:.6f}")
            logger.info(f"  KL penalty: {kl_penalty:.6f}")
        else:
            kl_div = torch.tensor(0.0, device=self.device)
            kl_penalty = torch.tensor(0.0, device=self.device)
            logger.info("  Reference model 없음, KL penalty = 0")
        
        # 엔트로피 보너스 계산 (CartPole GRPO 방식)
        # 현재 정책의 엔트로피 추정 (로그 확률의 분산 기반)
        entropy_estimate = current_log_probs.var()
        entropy_bonus = self.grpo_config.entropy_coef * entropy_estimate
        
        # 총 손실 (정책 손실 + KL 페널티 - 엔트로피 보너스)
        total_loss = policy_loss + kl_penalty - entropy_bonus
        
        logger.info(f"  Entropy estimate: {entropy_estimate:.6f}")
        logger.info(f"  Entropy bonus: {entropy_bonus:.6f}")
        logger.info(f"  Total loss: {total_loss:.6f}")
        
        # 최종 손실 안전성 검사
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error("🚨 총 손실에 nan/inf 발견! 학습 건너뛰기")
            return {
                'policy_loss': 0.0,
                'kl_div': 0.0,
                'total_loss': 0.0,
                'mean_reward': rewards.mean().item(),
                'baseline': baseline.item(),
                'mean_advantage': advantages.mean().item(),
                'error': 'nan_inf_in_loss'
            }
        
        # 역전파
        self.grpo_optimizer.zero_grad()
        total_loss.backward()
        
        # EasyR1 스타일 그래디언트 안정성 기법 적용
        self.training_step += 1
        
        # 1. 그래디언트 중앙화 (Gradient Centralization)
        if self.grpo_config.use_grad_centralization:
            self._apply_gradient_centralization()
        
        # 2. 그래디언트 정규화 (Adaptive Gradient Normalization)
        if self.grpo_config.use_grad_normalization:
            self._apply_gradient_normalization()
        
        # 3. 적응적 그래디언트 클리핑 (AdaGC)
        if self.grpo_config.use_adaptive_grad_clip:
            grad_norm = self._apply_adaptive_gradient_clipping()
        else:
            # 기본 그래디언트 클리핑
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        logger.info(f"🔧 그래디언트 처리:")
        logger.info(f"  - 그래디언트 norm: {grad_norm:.6f}")
        logger.info(f"  - 적응적 클리핑 임계값: {self.grad_norm_ema * self.grpo_config.grad_clip_coef:.6f}")
        
        self.grpo_optimizer.step()
        
        # 메트릭 저장 (메모리 정리 전에)
        metrics = {
            'policy_loss': policy_loss.item(),
            'kl_div': kl_div.item(),
            'entropy': entropy_estimate.item(),
            'entropy_bonus': entropy_bonus.item(),
            'total_loss': total_loss.item(),
            'mean_reward': sum(rewards) / len(rewards),
            'baseline': baseline,
            'mean_advantage': advantages.mean().item()
        }
        
        # 메모리 정리
        del current_log_probs, ref_log_probs, ratio, clipped_ratio
        del policy_loss, kl_penalty, total_loss, rewards, advantages
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return metrics
    
    def update_grpo_policy_multiple_epochs(self, experiences: List[Dict]) -> Dict:
        """다중 에포크 GRPO 업데이트 (CartPole GRPO 방식)"""
        if not experiences:
            return {}
        
        num_epochs = self.grpo_config.grpo_epochs
        logger.info(f"🔄 다중 에포크 GRPO 업데이트 시작 ({num_epochs} 에포크)")
        
        # 누적 메트릭
        total_policy_loss = 0.0
        total_kl_div = 0.0
        total_entropy = 0.0
        
        for epoch in range(num_epochs):
            logger.info(f"  에포크 {epoch + 1}/{num_epochs}")
            
            # 단일 에포크 업데이트
            metrics = self.update_grpo_policy(experiences)
            
            # 메트릭 누적
            total_policy_loss += metrics.get('policy_loss', 0.0)
            total_kl_div += metrics.get('kl_div', 0.0)
            total_entropy += metrics.get('entropy', 0.0)
            
            logger.info(f"    Policy loss: {metrics.get('policy_loss', 0.0):.6f}")
            logger.info(f"    KL div: {metrics.get('kl_div', 0.0):.6f}")
            logger.info(f"    Entropy: {metrics.get('entropy', 0.0):.6f}")
        
        # 평균 메트릭 계산
        avg_metrics = {
            'avg_policy_loss': total_policy_loss / num_epochs,
            'avg_kl_div': total_kl_div / num_epochs,
            'avg_entropy': total_entropy / num_epochs,
            'num_epochs': num_epochs,
            'total_experiences': len(experiences)
        }
        
        logger.info(f"✅ 다중 에포크 업데이트 완료")
        logger.info(f"  평균 Policy loss: {avg_metrics['avg_policy_loss']:.6f}")
        logger.info(f"  평균 KL div: {avg_metrics['avg_kl_div']:.6f}")
        
        return avg_metrics

    def _post_process_output(self, raw_output):
        """생성된 출력 후처리"""
        # 불필요한 텍스트 제거
        enhanced = raw_output.strip()
        
        # "Enhanced prompt:" 등의 레이블 제거
        enhanced = re.sub(r'^(Enhanced prompt:|Prompt:|Result:)\s*', '', enhanced, flags=re.IGNORECASE)
        enhanced = enhanced.strip()
        
        # 따옴표 제거
        enhanced = enhanced.strip('"\'')
        
        return enhanced
    
    def enhance_prompts_batch(self, user_prompts):
        """배치로 여러 프롬프트 향상"""
        results = []
        for prompt in user_prompts:
            result = self.enhance_prompt(prompt)
            results.append(result)
        return results

    def update_reference_model(self):
        """매 iteration마다 현재 모델을 reference로 복사 (CartPole GRPO 방식)"""
        if self.ref_model is not None:
            logger.info("🔄 Reference 모델 업데이트 중...")
            self.ref_model.load_state_dict(self.model.state_dict())
            self.ref_model.eval()
            logger.info("✅ Reference 모델 업데이트 완료")
        else:
            logger.warning("⚠️ Reference 모델이 없어 업데이트 건너뛰기")
    
    def move_ref_model_to_device(self, device: str):
        """Reference 모델을 지정된 디바이스로 이동 (전체 학습에서는 비활성화)"""
        if self.ref_model is not None:
            logger.info(f"🔧 Reference 모델을 {device}로 이동")
            self.ref_model = self.ref_model.to(device)
        else:
            logger.info("🎯 전체 학습 모드: Reference 모델 이동 건너뛰기")
    
    def save_lora_model(self, save_path: str):
        """LoRA 어댑터 저장"""
        try:
            self.model.save_pretrained(save_path)
            logger.info(f"✅ LoRA 모델 저장 완료: {save_path}")
        except Exception as e:
            logger.error(f"❌ LoRA 모델 저장 실패: {e}")
    
    def load_lora_model(self, load_path: str):
        """LoRA 어댑터 로드"""
        try:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, load_path)
            logger.info(f"✅ LoRA 모델 로드 완료: {load_path}")
        except Exception as e:
            logger.error(f"❌ LoRA 모델 로드 실패: {e}")
    
    def get_lora_trainable_params(self):
        """LoRA 학습 가능한 파라미터 정보 반환"""
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return {
            'trainable_params': trainable_params,
            'all_params': all_param,
            'trainable_percentage': 100 * trainable_params / all_param
        }
    
    def _apply_gradient_centralization(self):
        """그래디언트 중앙화 (EasyR1 스타일)"""
        for param in self.model.parameters():
            if param.grad is not None and param.grad.dim() > 1:
                # 그래디언트의 평균을 빼서 중앙화
                grad_mean = param.grad.mean(dim=tuple(range(1, param.grad.dim())), keepdim=True)
                param.grad = param.grad - grad_mean
    
    def _apply_gradient_normalization(self):
        """적응적 그래디언트 정규화 (EasyR1 스타일)"""
        alpha = self.grpo_config.grad_norm_alpha
        
        for param in self.model.parameters():
            if param.grad is not None:
                grad = param.grad
                grad_std = grad.std()
                
                # 표준편차가 0이 아닐 때만 정규화 적용
                if grad_std > 1e-8:
                    normalized_grad = grad / (grad_std + 1e-8)
                    param.grad = (1 - alpha) * grad + alpha * normalized_grad
    
    def _apply_adaptive_gradient_clipping(self) -> float:
        """적응적 그래디언트 클리핑 (AdaGC 스타일)"""
        # 현재 그래디언트 norm 계산
        grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # 지수 이동 평균 업데이트
        if self.grad_norm_ema == 0.0:
            self.grad_norm_ema = grad_norm
        else:
            beta = self.grpo_config.grad_clip_ema_beta
            self.grad_norm_ema = beta * self.grad_norm_ema + (1 - beta) * grad_norm
        
        # 적응적 클리핑 임계값 계산
        clip_threshold = self.grpo_config.grad_clip_coef * self.grad_norm_ema
        
        # 클리핑 적용
        if grad_norm > clip_threshold:
            clip_coef = clip_threshold / (grad_norm + 1e-8)
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
            
            logger.info(f"🔧 적응적 그래디언트 클리핑 적용: {grad_norm:.6f} -> {clip_threshold:.6f}")
        
        return grad_norm
