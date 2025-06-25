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
    """QWEN GRPO 통합 설정 (LoRA 최적화)"""
    learning_rate: float = 2e-4  # LoRA는 더 높은 학습률 사용 가능
    batch_size: int = 8  # LoRA로 메모리 절약되어 배치 크기 증가 가능
    num_rollouts: int = 6  # 더 많은 롤아웃 가능
    max_prompt_length: int = 77
    max_new_tokens: int = 30
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 100
    kl_coef: float = 0.02
    clip_ratio: float = 0.1
    entropy_coef: float = 0.02
    num_enhancement_candidates: int = 20  # 생성할 후보 개수
    save_images: bool = True
    log_dir: str = "grpo_results"
    # Semantic filtering 설정 (제거됨 - GRPO가 직접 학습하도록)
    use_semantic_filtering: bool = False  # 후보 생성시 semantic filtering 사용
    semantic_threshold: float = 0.7  # Semantic similarity threshold

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
        
        # LoRA 설정 및 적용 - 균형잡힌 설정
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # LoRA rank 복원 (성능과 메모리 균형)
            lora_alpha=32,  # LoRA scaling parameter 복원
            lora_dropout=0.1,
            target_modules=[
                # Attention 모듈들
                "q_proj", "v_proj", "k_proj", "o_proj",
                # MLP 모듈들 (선택적)
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
            inference_mode=False,
        )
        
        # LoRA 어댑터 적용
        self.model = get_peft_model(self.model, lora_config)
        logger.info("✅ LoRA 어댑터 적용 완료")
        
        # LoRA 파라미터 정보 출력
        trainable_params = self.model.get_nb_trainable_parameters()
        all_params = self.model.num_parameters()
        trainable_percentage = 100 * trainable_params / all_params
        
        logger.info(f"📊 LoRA 파라미터 정보:")
        logger.info(f"  - 학습 가능한 파라미터: {trainable_params:,}")
        logger.info(f"  - 전체 파라미터: {all_params:,}")
        logger.info(f"  - 학습 비율: {trainable_percentage:.2f}%")
        logger.info(f"  - LoRA rank: {lora_config.r}")
        logger.info(f"  - LoRA alpha: {lora_config.lora_alpha}")
        logger.info(f"  - 타겟 모듈: {lora_config.target_modules}")
        
        # 메모리 최적화 설정
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing 활성화")

        if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 생성 설정 - LoRA 최적화
        self.generation_config = GenerationConfig(
            max_new_tokens=30,  # 적절한 토큰 수로 복원
            temperature=self.temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=False,  # 메모리 절약을 위해 캐시 비활성화 유지
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
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        logger.info(f"📊 LoRA 학습 파라미터 개수: {len(trainable_params)}")
        
        self.grpo_optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=self.grpo_config.learning_rate,
            weight_decay=0.01
        )
        
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 GPU 메모리 캐시 정리 완료")
        
        logger.info("✅ GRPO 컴포넌트 초기화 완료 (메모리 최적화)")
        logger.info(f"📊 QWEN 모델 직접 학습 방식으로 설정")

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
        
        # 로그 확률 계산 (생성된 토큰들의 평균 로그 확률)
        if hasattr(outputs, 'scores') and outputs.scores:
            log_probs = []
            for i, score in enumerate(outputs.scores):
                token_id = outputs.sequences[0][inputs['input_ids'].shape[1] + i]
                log_prob = F.log_softmax(score, dim=-1)[0, token_id]
                log_probs.append(log_prob)
            
            avg_log_prob = torch.stack(log_probs).mean()
        else:
            # fallback: 더미 로그 확률
            if self.device == "accelerate":
                avg_log_prob = torch.tensor(0.0)  # Accelerate가 디바이스 관리
            else:
                avg_log_prob = torch.tensor(0.0, device=self.device)
        
        return enhanced_prompt, avg_log_prob

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
            
            # 생성된 부분의 로그 확률만 계산
            prompt_length = len(self.tokenizer.encode(prompt))
            generated_logits = outputs.logits[0, prompt_length-1:-1]  # 생성된 부분만
            generated_tokens = inputs['input_ids'][0, prompt_length:]
            
            log_probs = F.log_softmax(generated_logits, dim=-1)
            token_log_probs = log_probs.gather(1, generated_tokens.unsqueeze(1)).squeeze(1)
            
            # 결과를 main model device로 이동
            result = token_log_probs.mean().to(self.device)
            return result

    def update_grpo_policy(self, experiences: List[Dict]) -> Dict:
        """GRPO 정책 업데이트 (QWEN 직접 학습)"""
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
        
        # 텐서로 변환 (단일 GPU 모드)
        old_log_probs = torch.stack(old_log_probs).half()
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float16)
        
        # Group-relative advantage 사용 (main.py에서 계산됨) - 단일 GPU 모드
        advantages = []
        for exp in experiences:
            if 'group_advantage' in exp:
                advantages.append(exp['group_advantage'])
            else:
                # fallback: 기존 방식
                advantages.append(exp['reward'] - rewards.mean().item())
        
        advantages = torch.tensor(advantages, device=self.device, dtype=torch.float16)
        
        baseline = rewards.mean()  # 로깅용
        
        # 현재 모델과 참조 모델의 로그 확률 계산
        current_log_probs = []
        ref_log_probs = []
        
        for user_prompt, enhanced_prompt in zip(user_prompts, enhanced_prompts):
            # 현재 모델의 로그 확률 (gradient 계산)
            _, current_log_prob = self.generate_grpo_enhanced_prompt(user_prompt)
            current_log_probs.append(current_log_prob)
            
            # 참조 모델의 로그 확률 (없으면 더미값)
            ref_log_prob = self.get_ref_model_log_prob(user_prompt, enhanced_prompt)
            ref_log_probs.append(ref_log_prob)
        
        current_log_probs = torch.stack(current_log_probs).half()
        ref_log_probs = torch.stack(ref_log_probs).half()
        
        # 중요도 비율 계산
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # PPO 클립된 목적 함수
        clipped_ratio = torch.clamp(ratio, 1 - self.grpo_config.clip_ratio, 1 + self.grpo_config.clip_ratio)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # KL 발산 페널티 (단일 GPU 모드 - Reference model 항상 있음)
        if self.ref_model is not None:
            kl_div = (current_log_probs - ref_log_probs).mean()
            kl_penalty = self.grpo_config.kl_coef * kl_div
        else:
            kl_div = torch.tensor(0.0, device=self.device)
            kl_penalty = torch.tensor(0.0, device=self.device)
        
        # 총 손실 (엔트로피는 QWEN 자체에서 제공)
        total_loss = policy_loss + kl_penalty
        
        # 역전파
        self.grpo_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.grpo_optimizer.step()
        
        # 메트릭 반환
        return {
            'policy_loss': policy_loss.item(),
            'kl_div': kl_div.item(),
            'total_loss': total_loss.item(),
            'mean_reward': rewards.mean().item(),
            'baseline': baseline.item(),
            'mean_advantage': advantages.mean().item()
        }

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
