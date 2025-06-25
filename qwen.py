import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, GenerationConfig
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class QWENGRPOConfig:
    """QWEN GRPO 통합 설정"""
    learning_rate: float = 1e-6
    batch_size: int = 4
    num_rollouts: int = 5
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
        self.grpo_config = grpo_config or QWENGRPOConfig()

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
  
        model_kwargs = {
            'torch_dtype': torch.float16,
            'trust_remote_code': True,
            'low_cpu_mem_usage': True
        }

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
        ).to(self.device)

        if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 생성 설정
        self.generation_config = GenerationConfig(
            max_new_tokens=77,
            temperature=self.temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        logger.info("Model loaded")
        
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
        """GRPO 관련 컴포넌트 설정 - QWEN 직접 학습"""
        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = len(self.tokenizer.get_vocab())
        
        # 참조 모델 (frozen copy of QWEN for KL penalty)
        from copy import deepcopy
        self.ref_model = deepcopy(self.model)
        self.ref_model.eval()
        
        # 참조 모델의 모든 파라미터를 freeze
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # 옵티마이저 (QWEN 모델만 학습)
        self.grpo_optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.grpo_config.learning_rate,
            weight_decay=0.01
        )
        
        logger.info("✅ GRPO 컴포넌트 초기화 완료")
        logger.info(f"📊 QWEN 모델 직접 학습 방식으로 설정")

    def _init_grpo_weights(self):
        """GRPO 가중치 초기화 (QWEN 직접 학습 방식에서는 불필요)"""
        pass

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
        ).to(self.device)
        
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
                outputs = self.model.generate(
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
        ).to(self.device)
        
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
        ).to(self.device)
        
        # QWEN 모델로 직접 생성 (gradient 계산)
        outputs = self.model.generate(
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
            avg_log_prob = torch.tensor(0.0, device=self.device)
        
        return enhanced_prompt, avg_log_prob

    def get_ref_model_log_prob(self, user_prompt: str, enhanced_prompt: str) -> torch.Tensor:
        """참조 모델의 로그 확률 계산"""
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
        
        # 토크나이징
        inputs = self.tokenizer(
            full_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)
        
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
            
            return token_log_probs.mean()

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
        
        # 텐서로 변환
        old_log_probs = torch.stack(old_log_probs).half()
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float16)
        
        # 그룹 평균을 baseline으로 사용 (GRPO 방식)
        baseline = rewards.mean()
        advantages = rewards - baseline
        
        # 현재 모델과 참조 모델의 로그 확률 계산
        current_log_probs = []
        ref_log_probs = []
        
        for user_prompt, enhanced_prompt in zip(user_prompts, enhanced_prompts):
            # 현재 모델의 로그 확률 (gradient 계산)
            _, current_log_prob = self.generate_grpo_enhanced_prompt(user_prompt)
            current_log_probs.append(current_log_prob)
            
            # 참조 모델의 로그 확률
            ref_log_prob = self.get_ref_model_log_prob(user_prompt, enhanced_prompt)
            ref_log_probs.append(ref_log_prob)
        
        current_log_probs = torch.stack(current_log_probs).half()
        ref_log_probs = torch.stack(ref_log_probs).half()
        
        # 중요도 비율 계산
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # PPO 클립된 목적 함수
        clipped_ratio = torch.clamp(ratio, 1 - self.grpo_config.clip_ratio, 1 + self.grpo_config.clip_ratio)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # KL 발산 페널티
        kl_div = (current_log_probs - ref_log_probs).mean()
        kl_penalty = self.grpo_config.kl_coef * kl_div
        
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
