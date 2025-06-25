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
        """GRPO 관련 컴포넌트 설정"""
        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = len(self.tokenizer.get_vocab())
        
        # GRPO 정책 헤드 (기존 모델 위에 추가)
        self.grpo_policy_head = nn.Sequential(
            nn.Linear(self.hidden_size, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, self.grpo_config.num_enhancement_candidates)  # 후보 개수만큼
        ).to(self.device).half()
        
        # 가중치 초기화
        self._init_grpo_weights()
        
        # 참조 정책 (frozen copy)
        self.ref_policy_head = nn.Sequential(
            nn.Linear(self.hidden_size, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, self.grpo_config.num_enhancement_candidates)
        ).to(self.device).half()
        
        self.ref_policy_head.load_state_dict(self.grpo_policy_head.state_dict())
        self.ref_policy_head.eval()
        
        # 옵티마이저 (GRPO 헤드만 학습)
        self.grpo_optimizer = torch.optim.AdamW(
            self.grpo_policy_head.parameters(), 
            lr=self.grpo_config.learning_rate, 
            weight_decay=0.01
        )
        
        logger.info("✅ GRPO 컴포넌트 초기화 완료")
        logger.info(f"📊 Action Space: {self.grpo_config.num_enhancement_candidates} enhancement candidates")

    def _init_grpo_weights(self):
        """GRPO 가중치 초기화"""
        for layer in self.grpo_policy_head:
            if isinstance(layer, nn.Linear):
                gain = 0.02 if layer.out_features == self.grpo_config.num_enhancement_candidates else 0.1
                nn.init.xavier_normal_(layer.weight, gain=gain)
                nn.init.constant_(layer.bias, 0.0)

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
        """GRPO를 위한 상태 표현 생성"""
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
        
        # 히든 스테이트 추출
        with torch.no_grad():
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

    def get_grpo_action_and_log_prob(self, user_prompt: str) -> Tuple[int, torch.Tensor, List[str]]:
        """GRPO 액션 선택 및 로그 확률 계산"""
        # 상태 표현 생성
        state_repr = self.get_grpo_state_representation(user_prompt)
        
        # 후보 프롬프트들 생성 (semantic filtering 제거)
        candidates = self.generate_enhancement_candidates(
            user_prompt, 
            use_semantic_filtering=False,  # GRPO가 직접 학습하도록
            semantic_threshold=self.grpo_config.semantic_threshold
        )
        
        # 정책 로짓 계산
        policy_logits = self.grpo_policy_head(state_repr.half())
        
        # 온도 스케일링
        scaled_logits = policy_logits / self.grpo_config.temperature
        scaled_logits = torch.clamp(scaled_logits, min=-10, max=10)
        
        # 확률 분포 생성
        action_probs = F.softmax(scaled_logits, dim=-1)
        
        # 액션 샘플링
        try:
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)
        except ValueError:
            logger.warning("Invalid probability distribution, using uniform sampling")
            action = torch.randint(0, len(candidates), (1,)).to(self.device)
            action_log_prob = torch.log(torch.tensor(1.0 / len(candidates), device=self.device))
        
        return action.item(), action_log_prob, candidates

    def get_ref_policy_log_prob(self, user_prompt: str, action: int) -> torch.Tensor:
        """참조 정책의 로그 확률 계산"""
        state_repr = self.get_grpo_state_representation(user_prompt)
        
        with torch.no_grad():
            ref_logits = self.ref_policy_head(state_repr.half())
            scaled_logits = ref_logits / self.grpo_config.temperature
            ref_probs = F.softmax(scaled_logits, dim=-1)
            ref_log_prob = torch.log(ref_probs[action] + 1e-8)
        
        return ref_log_prob

    def update_grpo_policy(self, experiences: List[Dict]) -> Dict:
        """GRPO 정책 업데이트"""
        if not experiences:
            return {}
        
        # 경험 데이터 준비
        states = []
        actions = []
        log_probs = []
        rewards = []
        
        for exp in experiences:
            states.append(self.get_grpo_state_representation(exp['user_prompt']))
            actions.append(exp['action'])
            log_probs.append(exp['log_prob'])
            rewards.append(exp['reward'])
        
        # 텐서로 변환
        states = torch.stack(states).half()
        actions = torch.tensor(actions, device=self.device)
        old_log_probs = torch.stack(log_probs).half()
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float16)
        
        # 그룹 평균을 baseline으로 사용 (GRPO 방식)
        baseline = rewards.mean()
        advantages = rewards - baseline
        
        # 현재 정책의 로그 확률 계산
        current_logits = self.grpo_policy_head(states)
        scaled_logits = current_logits / self.grpo_config.temperature
        current_probs = F.softmax(scaled_logits, dim=-1)
        current_log_probs = torch.log(current_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
        
        # 참조 정책의 로그 확률 계산
        with torch.no_grad():
            ref_logits = self.ref_policy_head(states)
            ref_scaled_logits = ref_logits / self.grpo_config.temperature
            ref_probs = F.softmax(ref_scaled_logits, dim=-1)
            ref_log_probs = torch.log(ref_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
        
        # 중요도 비율 계산
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # PPO 클립된 목적 함수
        clipped_ratio = torch.clamp(ratio, 1 - self.grpo_config.clip_ratio, 1 + self.grpo_config.clip_ratio)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # KL 발산 페널티
        kl_div = (current_log_probs - ref_log_probs).mean()
        kl_penalty = self.grpo_config.kl_coef * kl_div
        
        # 엔트로피 보너스
        entropy = -(current_probs * torch.log(current_probs + 1e-8)).sum(dim=-1).mean()
        entropy_bonus = self.grpo_config.entropy_coef * entropy
        
        # 총 손실
        total_loss = policy_loss + kl_penalty - entropy_bonus
        
        # 역전파
        self.grpo_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.grpo_policy_head.parameters(), 1.0)
        self.grpo_optimizer.step()
        
        # 메트릭 반환
        return {
            'policy_loss': policy_loss.item(),
            'kl_div': kl_div.item(),
            'entropy': entropy.item(),
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
