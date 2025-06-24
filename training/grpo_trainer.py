"""
GRPO Trainer for QWEN Model
===========================

QWEN 모델을 GRPO 알고리즘으로 학습시키는 트레이너입니다.

핵심 아이디어:
- State: user_prompt + placeholder + 현재까지 생성된 토큰들
- Action: placeholder에 추가할 다음 단어/토큰 선택
- Environment: SD3로 이미지 생성
- Reward: CLIP(original_user_prompt, generated_image)
- Reference Model: 학습 전 QWEN 모델

기반 코드: grpo-cartpole.ipynb

Author: AI Assistant
Date: 2025-01-22
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import logging
import copy
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GRPOConfig:
    """GRPO 학습 설정"""
    # 학습 파라미터
    learning_rate: float = 1e-5
    group_size: int = 4              # 그룹당 rollout 수
    num_iterations: int = 100        # 전체 학습 iteration
    grpo_epochs: int = 3             # 각 iteration당 최적화 epoch
    
    # GRPO 하이퍼파라미터
    gamma: float = 0.99              # 할인 팩터
    grpo_kl_beta: float = 0.01       # KL 발산 페널티
    grpo_clip_epsilon: float = 0.2   # 클리핑 파라미터
    entropy_coeff: float = 0.01      # 엔트로피 보너스
    
    # 토큰 생성 파라미터
    max_new_tokens: int = 10         # placeholder에 추가할 최대 토큰 수
    temperature: float = 0.8         # 샘플링 온도
    
    # 시스템 설정
    device: str = "cuda"
    epsilon_std: float = 1e-8        # 수치 안정성

class PromptEnvironment:
    """
    프롬프트 생성 환경
    
    - State: user_prompt + placeholder + generated_tokens
    - Action: 다음 단어 선택
    - Reward: CLIP similarity
    """
    
    def __init__(self, qwen_model, sd3_generator, clip_calculator, config: GRPOConfig):
        self.qwen_model = qwen_model
        self.sd3_generator = sd3_generator
        self.clip_calculator = clip_calculator
        self.config = config
        
        # 어휘 설정
        self.tokenizer = qwen_model.tokenizer
        self.vocab_size = len(self.tokenizer)
        
        # 자유로운 토큰 생성을 위한 어휘 설정
        # 전체 vocabulary에서 일부를 선택 (너무 크면 메모리 문제)
        self.vocab_size = len(self.tokenizer)
        
        # 유용한 토큰들을 필터링 (특수 토큰, 너무 짧은 토큰 제외)
        self.useful_token_ids = []
        for token_id in range(min(10000, self.vocab_size)):  # 처음 10k 토큰만 사용
            token_text = self.tokenizer.decode([token_id]).strip()
            
            # 필터링 조건
            if (len(token_text) >= 2 and  # 2글자 이상
                token_text.isalpha() and  # 알파벳만
                not token_text.startswith('<') and  # 특수 토큰 제외
                not token_text.startswith('[') and
                token_id not in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]):
                self.useful_token_ids.append(token_id)
        
        self.action_space_size = len(self.useful_token_ids)
        
        logger.info(f"🎮 Environment initialized with {self.action_space_size} useful tokens (unrestricted vocabulary)")
    
    def reset(self, user_prompt: str) -> torch.Tensor:
        """
        환경 리셋 - 새로운 user prompt로 시작
        
        Returns:
            초기 state (user_prompt + base_placeholder의 임베딩)
        """
        self.user_prompt = user_prompt
        
        # 기본 placeholder 추가
        base_placeholder = ", high quality, detailed"
        self.current_prompt = user_prompt + base_placeholder
        
        # 현재 상태를 토큰 ID로 변환 (CPU에서 처리)
        self.current_token_ids = self.tokenizer.encode(
            self.current_prompt, 
            add_special_tokens=False,
            return_tensors="pt"
        ).squeeze(0)
        
        # State는 현재 프롬프트의 마지막 몇 토큰의 임베딩
        state = self._get_state()
        
        return state
    
    def _get_state(self) -> torch.Tensor:
        """
        현재 상태 반환 (현재 프롬프트의 임베딩)
        """
        # 마지막 몇 토큰만 사용 (메모리 효율성)
        max_state_tokens = 20
        if len(self.current_token_ids) > max_state_tokens:
            state_token_ids = self.current_token_ids[-max_state_tokens:]
        else:
            state_token_ids = self.current_token_ids
        
        # 토큰 ID를 올바른 장치로 이동
        device = next(self.qwen_model.model.parameters()).device
        state_token_ids = state_token_ids.to(device)
        
        # 토큰 임베딩으로 변환
        with torch.no_grad():
            embeddings = self.qwen_model.model.get_input_embeddings()(state_token_ids.unsqueeze(0))
            # 평균 풀링으로 고정 크기 state 생성
            state = embeddings.mean(dim=1).squeeze(0)  # [hidden_size]
        
        return state
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        """
        액션 실행
        
        Args:
            action: 선택된 품질 토큰 인덱스
            
        Returns:
            next_state, reward, done
        """
        # 액션을 토큰 ID로 변환
        if action < len(self.useful_token_ids):
            selected_token_id = self.useful_token_ids[action]
        else:
            selected_token_id = self.useful_token_ids[0]  # 폴백
        
        # 토큰을 텍스트로 변환하여 프롬프트에 추가
        selected_token_text = self.tokenizer.decode([selected_token_id])
        self.current_prompt += " " + selected_token_text.strip()
        
        # 토큰 ID 업데이트 (CPU에서 처리)
        self.current_token_ids = self.tokenizer.encode(
            self.current_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).squeeze(0)
        
        # 새로운 상태 계산
        next_state = self._get_state()
        
        # 에피소드 종료 조건
        current_new_tokens = len(self.current_token_ids) - len(self.tokenizer.encode(
            self.user_prompt, add_special_tokens=False
        ))
        done = current_new_tokens >= self.config.max_new_tokens
        
        # 보상 계산 (에피소드 끝에서만)
        if done:
            reward = self._calculate_reward()
        else:
            reward = 0.0  # 중간 스텝에서는 보상 없음
        
        return next_state, reward, done
    
    def _calculate_reward(self) -> float:
        """
        CLIP을 사용한 보상 계산
        
        중요: 원본 user_prompt와 생성된 이미지 간의 유사도만 계산!
        """
        try:
            # SD3로 이미지 생성 (현재 향상된 프롬프트 사용)
            image = self.sd3_generator.generate_image(self.current_prompt)
            
            if image is None:
                return 0.0
            
            # CLIP 보상 계산 (원본 user_prompt 사용!)
            reward = self.clip_calculator.calculate_reward(self.user_prompt, image)
            
            logger.debug(f"🎯 Reward: {reward:.3f} for '{self.user_prompt}' -> '{self.current_prompt[:50]}...'")
            return reward
            
        except Exception as e:
            logger.warning(f"⚠️ Reward calculation failed: {e}")
            return 0.0
    
    def get_action_space_size(self) -> int:
        return self.action_space_size
    
    def get_state_dimension(self) -> int:
        return self.qwen_model.model.config.hidden_size

class GRPOTrainer:
    """
    QWEN 모델을 위한 GRPO 트레이너
    
    CartPole GRPO 코드를 기반으로 텍스트 생성에 적용
    """
    
    def __init__(self, qwen_model, sd3_generator, clip_calculator, config: GRPOConfig):
        self.qwen_model = qwen_model  # 학습할 정책 네트워크
        self.sd3_generator = sd3_generator
        self.clip_calculator = clip_calculator
        self.config = config
        
        # 환경 초기화
        self.env = PromptEnvironment(qwen_model, sd3_generator, clip_calculator, config)
        
        # 디바이스 설정
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        # 정책 네트워크 (QWEN 모델을 액션 확률 분포 출력하도록 어댑터 추가)
        self.action_head = nn.Linear(
            self.env.get_state_dimension(),
            self.env.get_action_space_size()
        ).to(self.device)
        
        # 옵티마이저 (QWEN 모델 + 액션 헤드 함께 학습)
        self.optimizer = optim.Adam(
            list(self.qwen_model.model.parameters()) + list(self.action_head.parameters()),
            lr=config.learning_rate
        )
        
        # 학습 통계
        self.iteration_rewards = []
        self.iteration_policy_losses = []
        self.iteration_entropies = []
        self.iteration_kl_divs = []
        
        logger.info(f"🚀 GRPO Trainer initialized for QWEN model")
    
    def get_action_distribution(self, state: torch.Tensor) -> Categorical:
        """
        상태에서 액션 확률 분포 계산
        """
        # QWEN 모델의 hidden state에서 액션 로짓 계산
        action_logits = self.action_head(state)
        return Categorical(logits=action_logits)
    
    def collect_group_trajectories(self, user_prompts: List[str]) -> Dict[str, Any]:
        """
        그룹 궤적 수집 (CartPole 코드 기반)
        """
        logger.info(f"🔄 Collecting trajectories for {len(user_prompts)} prompts...")
        
        group_states_list = []
        group_actions_list = []
        group_log_probs_old_list = []
        group_rewards_list = []
        
        episode_rewards_in_iter = []
        episode_lengths_in_iter = []
        
        # 평가 모드로 rollout
        self.qwen_model.model.eval()
        self.action_head.eval()
        
        for rollout_idx, user_prompt in enumerate(user_prompts):
            rollout_states = []
            rollout_actions = []
            rollout_log_probs = []
            rollout_rewards = []
            
            # 환경 리셋
            state = self.env.reset(user_prompt)
            episode_reward = 0.0
            episode_steps = 0
            done = False
            
            while not done and episode_steps < self.config.max_new_tokens:
                state_tensor = state.to(self.device)
                
                # 액션 선택
                with torch.no_grad():
                    action_dist = self.get_action_distribution(state_tensor)
                    action_tensor = action_dist.sample()
                    log_prob = action_dist.log_prob(action_tensor)
                
                # 환경에서 스텝 실행
                next_state, reward, done = self.env.step(action_tensor.item())
                
                # 데이터 저장
                rollout_states.append(state_tensor)
                rollout_actions.append(action_tensor)
                rollout_log_probs.append(log_prob)
                rollout_rewards.append(reward)
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
            
            # 완료된 rollout 저장
            if rollout_states:
                group_states_list.append(torch.stack(rollout_states))
                group_actions_list.append(torch.stack(rollout_actions).squeeze())
                group_log_probs_old_list.append(torch.stack(rollout_log_probs).squeeze())
                group_rewards_list.append(rollout_rewards)
            
            episode_rewards_in_iter.append(episode_reward)
            episode_lengths_in_iter.append(episode_steps)
        
        # 훈련 모드로 복원
        self.qwen_model.model.train()
        self.action_head.train()
        
        return {
            'states': group_states_list,
            'actions': group_actions_list,
            'log_probs_old': group_log_probs_old_list,
            'rewards': group_rewards_list,
            'episode_rewards': episode_rewards_in_iter,
            'episode_lengths': episode_lengths_in_iter
        }
    
    def calculate_group_advantages(self, group_data: Dict[str, Any]) -> List[torch.Tensor]:
        """
        그룹 상대적 어드밴티지 계산 (CartPole 코드 기반)
        """
        group_rewards_list = group_data['rewards']
        
        # 1단계: 할인된 리턴 계산
        all_raw_advantages = []
        temp_raw_advantages_tensors = []
        
        for rollout_rewards in group_rewards_list:
            rollout_len = len(rollout_rewards)
            rollout_advantages = torch.zeros(rollout_len, dtype=torch.float32, device=self.device)
            
            if rollout_len > 0:
                # 할인된 리턴 계산
                discounted_return = 0.0
                for t in reversed(range(rollout_len)):
                    discounted_return = rollout_rewards[t] + self.config.gamma * discounted_return
                    rollout_advantages[t] = discounted_return
                
                temp_raw_advantages_tensors.append(rollout_advantages)
                all_raw_advantages.extend(rollout_advantages.cpu().numpy())
            else:
                temp_raw_advantages_tensors.append(torch.empty((0,), device=self.device))
        
        # 2단계: 그룹 정규화
        if len(all_raw_advantages) > 1:
            group_mean = np.mean(all_raw_advantages)
            group_std = np.std(all_raw_advantages)
        else:
            group_mean = 0.0
            group_std = 1.0
        
        # 3단계: 정규화된 어드밴티지 계산
        group_advantages_list = []
        for raw_advantages in temp_raw_advantages_tensors:
            if raw_advantages.nelement() > 0:
                normalized_advantages = (raw_advantages - group_mean) / (group_std + self.config.epsilon_std)
            else:
                normalized_advantages = raw_advantages
            group_advantages_list.append(normalized_advantages)
        
        return group_advantages_list
    
    def update_grpo(self, group_data: Dict[str, Any], group_advantages: List[torch.Tensor]) -> Tuple[float, float, float]:
        """
        GRPO 업데이트 (CartPole 코드 기반)
        """
        # 데이터 연결
        states = torch.cat(group_data['states'], dim=0).to(self.device)
        actions = torch.cat(group_data['actions'], dim=0).to(self.device)
        log_probs_old = torch.cat(group_data['log_probs_old'], dim=0).to(self.device)
        advantages = torch.cat(group_advantages, dim=0).to(self.device)
        
        # 상수로 고정
        advantages = advantages.detach()
        log_probs_old = log_probs_old.detach()
        
        # 참조 모델 생성 (현재 모델 상태 복사)
        action_head_ref = copy.deepcopy(self.action_head)
        action_head_ref.eval()
        
        total_policy_objective = 0.0
        total_kl_div = 0.0
        total_entropy = 0.0
        
        for epoch in range(self.config.grpo_epochs):
            # 현재 정책으로 확률 계산
            action_dist = self.get_action_distribution(states)
            log_probs_new = action_dist.log_prob(actions)
            entropy = action_dist.entropy().mean()
            
            # 비율 계산
            ratio = torch.exp(log_probs_new - log_probs_old)
            
            # 클리핑된 서로게이트 목적 함수
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.config.grpo_clip_epsilon, 1.0 + self.config.grpo_clip_epsilon) * advantages
            clipped_surrogate_objective = torch.min(surr1, surr2).mean()
            
            # KL 발산 계산
            with torch.no_grad():
                action_dist_ref = Categorical(logits=action_head_ref(states))
                log_probs_ref = action_dist_ref.log_prob(actions)
            
            log_ratio_ref_curr = log_probs_ref - log_probs_new.detach()
            kl_div_estimate = torch.exp(log_ratio_ref_curr) - log_ratio_ref_curr - 1
            kl_div_estimate_mean = torch.relu(kl_div_estimate.mean())
            
            # 전체 손실
            policy_loss = -clipped_surrogate_objective + self.config.grpo_kl_beta * kl_div_estimate_mean - self.config.entropy_coeff * entropy
            
            # 최적화
            self.optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.qwen_model.model.parameters()) + list(self.action_head.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()
            
            # 통계 수집
            total_policy_objective += clipped_surrogate_objective.item()
            total_kl_div += kl_div_estimate_mean.item()
            total_entropy += entropy.item()
        
        # 평균 계산
        avg_policy_objective = total_policy_objective / self.config.grpo_epochs
        avg_kl_div = total_kl_div / self.config.grpo_epochs
        avg_entropy = total_entropy / self.config.grpo_epochs
        
        return avg_policy_objective, avg_kl_div, avg_entropy
    
    def train_iteration(self, user_prompts: List[str]) -> Dict[str, float]:
        """
        하나의 GRPO 학습 iteration 실행
        """
        # 1. 궤적 수집
        group_data = self.collect_group_trajectories(user_prompts)
        
        # 2. 어드밴티지 계산
        group_advantages = self.calculate_group_advantages(group_data)
        
        # 3. GRPO 업데이트
        avg_policy_obj, avg_kl, avg_entropy = self.update_grpo(group_data, group_advantages)
        
        # 4. 통계 기록
        avg_reward = np.mean(group_data['episode_rewards']) if group_data['episode_rewards'] else 0.0
        avg_length = np.mean(group_data['episode_lengths']) if group_data['episode_lengths'] else 0.0
        
        self.iteration_rewards.append(avg_reward)
        self.iteration_policy_losses.append(avg_policy_obj)
        self.iteration_entropies.append(avg_entropy)
        self.iteration_kl_divs.append(avg_kl)
        
        return {
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'policy_objective': avg_policy_obj,
            'kl_divergence': avg_kl,
            'entropy': avg_entropy
        } 