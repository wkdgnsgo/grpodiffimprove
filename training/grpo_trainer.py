"""
Enhanced GRPO Trainer for QWEN Model (Based on EasyR1)
======================================================

EasyR1의 verl/trainer/core_algos.py를 기반으로 한 완전히 새로운 GRPO 구현

핵심 특징:
- 정확한 GRPO advantage 계산 (그룹 내 정규화)
- Adaptive/Fixed KL Controller
- 이중 클리핑 policy loss
- 수치적 안정성 보장

Author: AI Assistant
Date: 2025-01-22
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Dict, Tuple, Optional, Any, Literal
import numpy as np
import logging
import copy
import os
import json
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# =============================================================================
# KL Controllers (from EasyR1)
# =============================================================================

class KLController(ABC):
    """KL coefficient controller base class"""
    def __init__(self):
        self.kl_coef = 0.02
    
    @abstractmethod
    def update(self, current_kl: float, n_steps: int):
        """Update kl_coef according to current KL."""
        pass

class AdaptiveKLController(KLController):
    """Adaptive KL controller from EasyR1"""
    
    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        super().__init__()
        self.kl_coef = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= mult

class FixedKLController(KLController):
    """Fixed KL controller from EasyR1"""
    
    def __init__(self, init_kl_coef: float):
        super().__init__()
        self.kl_coef = init_kl_coef

    def update(self, current_kl: float, n_steps: int):
        pass

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GRPOConfig:
    """EasyR1 기반 GRPO 설정"""
    # 기본 학습 설정
    learning_rate: float = 1e-6
    group_size: int = 4
    num_iterations: int = 50
    grpo_epochs: int = 3
    max_new_tokens: int = 10
    
    # KL 제어
    kl_type: str = "adaptive"  # "adaptive" or "fixed"
    kl_coef: float = 0.02
    kl_target: float = 0.01
    kl_horizon: float = 1000
    kl_penalty: str = "kl"  # "kl", "abs", "mse", "low_var_kl", "full"
    
    # Policy 클리핑
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    clip_ratio_dual: float = 4.0
    
    # 기타
    gamma: float = 0.99
    entropy_coeff: float = 0.01
    max_grad_norm: float = 1.0
    temperature: float = 0.8
    
    # 시스템
    device: str = "cuda"
    epsilon_std: float = 1e-6
    loss_avg_mode: str = "token"
    
    # 저장
    save_training_data: bool = True
    save_dir: str = "training_results"

# =============================================================================
# Utility Functions (from EasyR1)
# =============================================================================

def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute masked mean"""
    return (values * mask).sum() / (mask.sum() + eps)

def masked_whiten(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Whiten values with mask"""
    values_masked = values * mask
    mean = masked_mean(values_masked, mask, eps)
    var = masked_mean((values_masked - mean) ** 2, mask, eps)
    return (values_masked - mean) / (var.sqrt() + eps)

def average_loss(values: torch.Tensor, mask: torch.Tensor, mode: str = "token", eps: float = 1e-8) -> torch.Tensor:
    """Average the loss"""
    if mode == "token":
        return masked_mean(values, mask, eps=eps)
    elif mode == "seq":
        return ((values * mask).sum(-1) / (mask.sum(-1) + eps)).mean()
    else:
        raise NotImplementedError(f"Unknown mode: {mode}.")

# =============================================================================
# GRPO Advantage Computation (from EasyR1)
# =============================================================================

@torch.no_grad()
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, 
    response_mask: torch.Tensor, 
    index: torch.Tensor, 
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    EasyR1의 GRPO advantage 계산
    같은 프롬프트의 여러 rollout을 그룹화해서 정규화
    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i].item()].append(scores[i])

    for idx in id2score:
        if len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            id2std[idx] = torch.std(torch.stack(id2score[idx]))
        else:
            # 단일 샘플인 경우
            id2mean[idx] = scores[0].clone()
            id2std[idx] = torch.tensor(1.0, device=scores.device)

    for i in range(bsz):
        if len(id2score[index[i].item()]) > 1:
            scores[i] = (scores[i] - id2mean[index[i].item()]) / (id2std[index[i].item()] + eps)
        else:
            scores[i] = 0.0

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns

# =============================================================================
# Policy Loss Computation (from EasyR1)
# =============================================================================

def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_dual: float,
    loss_avg_mode: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """EasyR1의 이중 클리핑 policy loss 계산"""
    
    negative_approx_kl = log_probs - old_log_probs
    negative_approx_kl = torch.clamp(negative_approx_kl, -20.0, 20.0)
    ratio = torch.exp(negative_approx_kl)
    
    clipped_ratio = torch.exp(
        torch.clamp(
            negative_approx_kl, 
            np.log(1.0 - clip_ratio_low), 
            np.log(1.0 + clip_ratio_high)
        )
    )

    # Metrics
    metrics = {"ppo_kl": masked_mean(-negative_approx_kl, response_mask).item()}
    metrics["entropy_loss"] = average_loss(-log_probs, response_mask, mode=loss_avg_mode).item()

    # 3가지 loss
    pg_loss = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    pg_loss3 = -advantages * clip_ratio_dual

    # 이중 클리핑
    clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)
    metrics["pg_clipfrac_higher"] = masked_mean((pg_loss < pg_loss2).float(), response_mask).item()
    
    clipped_pg_loss_lower = torch.min(clipped_pg_loss_higher, pg_loss3)
    final_pg_loss = torch.where(advantages < 0, clipped_pg_loss_lower, clipped_pg_loss_higher)
    metrics["pg_clipfrac_lower"] = masked_mean(
        (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float(), 
        response_mask
    ).item()

    final_pg_loss = average_loss(final_pg_loss, response_mask, mode=loss_avg_mode)
    
    return final_pg_loss, metrics

# =============================================================================
# KL Divergence Computation (from EasyR1)
# =============================================================================

def compute_kl(
    log_probs: torch.Tensor, 
    ref_log_probs: torch.Tensor, 
    kl_penalty: str
) -> torch.Tensor:
    """EasyR1의 KL divergence 계산"""
    log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
    
    if kl_penalty == "kl":
        return log_probs - ref_log_probs
    elif kl_penalty == "abs":
        return (log_probs - ref_log_probs).abs()
    elif kl_penalty == "mse":
        return 0.5 * (log_probs - ref_log_probs).square()
    elif kl_penalty == "low_var_kl":
        kl = (ref_log_probs - log_probs).clamp(-20.0, 20.0)
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10.0, max=10.0)
    elif kl_penalty == "full":
        return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)
    else:
        raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}.")

# =============================================================================
# Environment
# =============================================================================

class PromptEnvironment:
    """프롬프트 생성 환경 (EasyR1 기반)"""
    
    def __init__(self, qwen_model, sd3_generator, clip_calculator, config: GRPOConfig):
        self.qwen_model = qwen_model
        self.sd3_generator = sd3_generator
        self.clip_calculator = clip_calculator
        self.config = config
        
        # 토크나이저 설정
        self.tokenizer = qwen_model.tokenizer
        
        # 유용한 토큰들 필터링
        self.useful_token_ids = []
        for token_id in range(min(10000, len(self.tokenizer))):
            token_text = self.tokenizer.decode([token_id]).strip()
            
            if (len(token_text) >= 2 and  
                token_text.isalpha() and  
                not token_text.startswith('<') and  
                not token_text.startswith('[') and
                token_id not in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]):
                self.useful_token_ids.append(token_id)
        
        self.action_space_size = len(self.useful_token_ids)
        
        # 상태 변수
        self.current_iteration = 0
        self.current_rollout = 0
        self.current_step = 0
        
        logger.info(f"🎮 Environment initialized with {self.action_space_size} tokens")
    
    def set_iteration_info(self, iteration: int, rollout: int):
        self.current_iteration = iteration
        self.current_rollout = rollout
        self.current_step = 0
    
    def reset(self, user_prompt: str) -> torch.Tensor:
        self.user_prompt = user_prompt
        
        # QwenWrapper의 enhance_prompt 메서드 사용
        try:
            # QWEN VL 모델로 프롬프트 향상
            enhancement_result = self.qwen_model.enhance_prompt(user_prompt)
            self.current_prompt = enhancement_result['enhanced_prompt']
            logger.info(f"🎯 QWEN enhanced: '{user_prompt}' -> '{self.current_prompt}'")
        except Exception as e:
            # 실패 시 fallback: 기본 placeholder 추가
            fallback_placeholder = ", high quality, detailed, professional photography"
            self.current_prompt = user_prompt + fallback_placeholder
            logger.warning(f"⚠️ QWEN enhancement failed, using fallback: {e}")
        
        self.current_token_ids = self.tokenizer.encode(
            self.current_prompt, 
            add_special_tokens=False,
            return_tensors="pt"
        ).squeeze(0)
        
        self.current_step = 0
        logger.info(f"🔄 Environment reset completed")
        return self._get_state()
    
    def _get_state(self) -> torch.Tensor:
        max_tokens = 20
        if len(self.current_token_ids) > max_tokens:
            state_token_ids = self.current_token_ids[-max_tokens:]
        else:
            state_token_ids = self.current_token_ids
        
        device = next(self.qwen_model.model.parameters()).device
        state_token_ids = state_token_ids.to(device)
        
        with torch.no_grad():
            embeddings = self.qwen_model.model.get_input_embeddings()(state_token_ids.unsqueeze(0))
            state = embeddings.mean(dim=1).squeeze(0)
            
            # 더 안정적인 정규화
            state_norm = state.norm(dim=-1, keepdim=True)
            # 매우 작은 norm 방지
            state_norm = torch.clamp(state_norm, min=self.config.epsilon_std)
            state = state / state_norm
            
            # NaN/Inf 체크
            if torch.isnan(state).any() or torch.isinf(state).any():
                logger.warning("⚠️ NaN/Inf in state embedding, using zero state")
                state = torch.zeros_like(state)
        
        return state
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        if action < len(self.useful_token_ids):
            selected_token_id = self.useful_token_ids[action]
        else:
            selected_token_id = self.useful_token_ids[0]
        
        selected_token_text = self.tokenizer.decode([selected_token_id])
        self.current_prompt += " " + selected_token_text.strip()
        
        self.current_token_ids = self.tokenizer.encode(
            self.current_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).squeeze(0)
        
        next_state = self._get_state()
        
        current_new_tokens = len(self.current_token_ids) - len(self.tokenizer.encode(
            self.user_prompt, add_special_tokens=False
        ))
        done = current_new_tokens >= self.config.max_new_tokens
        
        if done:
            reward = self._calculate_reward()
        else:
            reward = 0.0
        
        self.current_step += 1
        return next_state, reward, done
    
    def _calculate_reward(self) -> float:
        try:
            enhanced_image = self.sd3_generator.generate_image(self.current_prompt)
            
            if enhanced_image is not None:
                reward = self.clip_calculator.calculate_reward(self.user_prompt, enhanced_image)
            else:
                reward = 0.0
            
            self._save_step_data(enhanced_image, reward)
            return reward
            
        except Exception as e:
            logger.warning(f"⚠️ Reward calculation failed: {e}")
            return 0.0
    
    def _save_step_data(self, enhanced_image, reward: float):
        if not self.config.save_training_data:
            return
        
        try:
            step_dir = os.path.join(
                self.config.save_dir,
                f"iteration_{self.current_iteration:03d}",
                f"rollout_{self.current_rollout:02d}",
                f"step_{self.current_step:02d}"
            )
            os.makedirs(step_dir, exist_ok=True)
            
            # 프롬프트 저장
            with open(os.path.join(step_dir, "original_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(self.user_prompt)
            
            with open(os.path.join(step_dir, "enhanced_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(self.current_prompt)
            
            # 이미지 저장
            if enhanced_image is not None:
                enhanced_image.save(os.path.join(step_dir, "enhanced_image.png"))
            
            # 메타데이터 저장
            metadata = {
                "iteration": self.current_iteration,
                "rollout": self.current_rollout,
                "step": self.current_step,
                "original_prompt": self.user_prompt,
                "enhanced_prompt": self.current_prompt,
                "reward": reward,
                "timestamp": datetime.now().isoformat(),
                "tokens_added": len(self.current_token_ids) - len(self.tokenizer.encode(self.user_prompt, add_special_tokens=False))
            }
            
            with open(os.path.join(step_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save step data: {e}")
    
    def get_action_space_size(self) -> int:
        return self.action_space_size
    
    def get_state_dimension(self) -> int:
        return self.qwen_model.model.config.hidden_size

# =============================================================================
# GRPO Trainer
# =============================================================================

class GRPOTrainer:
    """EasyR1 기반 GRPO 트레이너"""
    
    def __init__(self, qwen_model, sd3_generator, clip_calculator, config: GRPOConfig):
        self.qwen_model = qwen_model
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
        
        # 액션 헤드 (더 안정적인 multi-layer 구조)
        qwen_dtype = next(self.qwen_model.model.parameters()).dtype
        hidden_size = self.env.get_state_dimension()
        action_size = self.env.get_action_space_size()
        
        logger.info(f"🎯 Action head: {hidden_size} -> {action_size} ({qwen_dtype})")
        
        # Multi-layer action head로 안정성 확보
        intermediate_size = min(512, action_size)  # 중간 크기 제한
        
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, dtype=qwen_dtype),
            nn.LayerNorm(intermediate_size, dtype=qwen_dtype),
            nn.Tanh(),  # 안정적인 활성화 함수
            nn.Linear(intermediate_size, action_size, dtype=qwen_dtype)
        ).to(self.device)
        
        # 더 보수적인 가중치 초기화
        with torch.no_grad():
            for module in self.action_head:
                if isinstance(module, nn.Linear):
                    # 매우 작은 초기화
                    nn.init.xavier_normal_(module.weight, gain=0.01)
                    nn.init.zeros_(module.bias)
                    # 추가 스케일링
                    module.weight.mul_(0.01)
        
        logger.info(f"🔧 Multi-layer action head initialized: {hidden_size}->{intermediate_size}->{action_size}")
        
        # 옵티마이저 (더 낮은 학습률)
        action_head_params = list(self.action_head.parameters())
        qwen_params = list(self.qwen_model.model.parameters())
        
        # Action head는 더 낮은 학습률 적용
        self.optimizer = optim.AdamW([
            {'params': qwen_params, 'lr': config.learning_rate},
            {'params': action_head_params, 'lr': config.learning_rate * 0.1}  # 10배 낮은 학습률
        ], eps=1e-8, weight_decay=1e-6, betas=(0.9, 0.95))
        
        # KL Controller
        self.kl_controller = self._get_kl_controller()
        
        # Reference model
        self.ref_model = copy.deepcopy(self.qwen_model.model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # 통계
        self.iteration_rewards = []
        self.iteration_policy_losses = []
        self.iteration_entropies = []
        self.iteration_kl_divs = []
        self.current_iteration = 0
        
        logger.info(f"🚀 GRPO Trainer initialized (KL: {config.kl_type})")
    
    def _get_kl_controller(self) -> KLController:
        if self.config.kl_type == "fixed":
            return FixedKLController(init_kl_coef=self.config.kl_coef)
        elif self.config.kl_type == "adaptive":
            return AdaptiveKLController(
                init_kl_coef=self.config.kl_coef,
                target_kl=self.config.kl_target,
                horizon=self.config.kl_horizon,
            )
        else:
            raise ValueError(f"Unknown kl type: {self.config.kl_type}.")
    
    def get_action_distribution(self, state: torch.Tensor) -> Categorical:
        # State 검증 및 정리
        if torch.isnan(state).any() or torch.isinf(state).any():
            logger.warning("⚠️ NaN/Inf in state, using zero state")
            state = torch.zeros_like(state)
        
        # dtype 일치 확인
        state = state.to(dtype=self.action_head[0].weight.dtype, device=self.device)
        
        # Forward pass
        if self.action_head.training:
            action_logits = self.action_head(state)
        else:
            with torch.no_grad():
                action_logits = self.action_head(state)
        
        # Logits 검증 및 처리
        if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
            logger.warning("⚠️ NaN/Inf in logits, using uniform distribution")
            action_logits = torch.zeros_like(action_logits)
        
        # 더 강한 클리핑 (안정성 확보)
        action_logits = torch.clamp(action_logits, min=-10.0, max=10.0)
        
        # Temperature scaling 적용
        action_logits = action_logits / self.config.temperature
        
        return Categorical(logits=action_logits)
    
    def collect_group_trajectories(self, user_prompts: List[str]) -> Dict[str, Any]:
        """그룹 궤적 수집"""
        logger.info(f"🔄 Collecting trajectories for {len(user_prompts)} prompts...")
        
        group_states_list = []
        group_actions_list = []
        group_log_probs_old_list = []
        group_rewards_list = []
        
        episode_rewards = []
        episode_lengths = []
        
        # 평가 모드
        self.qwen_model.model.eval()
        self.action_head.eval()
        
        for rollout_idx, user_prompt in enumerate(user_prompts):
            rollout_states = []
            rollout_actions = []
            rollout_log_probs = []
            rollout_rewards = []
            
            # iteration 정보 설정
            self.env.set_iteration_info(self.current_iteration, rollout_idx)
            
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
                
                # 환경 스텝
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
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
        
        # 훈련 모드 복원
        self.qwen_model.model.train()
        self.action_head.train()
        
        return {
            'states': group_states_list,
            'actions': group_actions_list,
            'log_probs_old': group_log_probs_old_list,
            'rewards': group_rewards_list,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
    
    def train_iteration(self, user_prompts: List[str]) -> Dict[str, float]:
        """단일 iteration 학습"""
        
        # 1. 궤적 수집
        group_data = self.collect_group_trajectories(user_prompts)
        
        if not group_data['states']:
            logger.warning("No valid trajectories collected")
            return {}
        
        # 2. GRPO advantage 계산
        advantages_list = []
        for i, rewards in enumerate(group_data['rewards']):
            # 토큰 레벨 보상 생성
            seq_len = len(group_data['states'][i])
            token_rewards = torch.zeros(1, seq_len, device=self.device)
            if rewards:
                token_rewards[0, -1] = rewards[-1]  # 마지막 토큰에만 보상
            
            # 응답 마스크 생성
            response_mask = torch.ones(1, seq_len, device=self.device)
            
            # 인덱스 (프롬프트별 그룹화용)
            index = torch.tensor([i % len(user_prompts)], device=self.device)
            
            # GRPO advantage 계산
            advantages, returns = compute_grpo_outcome_advantage(
                token_rewards, response_mask, index
            )
            # Advantage는 gradient가 필요 없음
            advantages_list.append(advantages.squeeze(0).detach())
        
        # 3. Policy 업데이트
        total_policy_loss = 0
        total_metrics = defaultdict(float)
        
        for epoch in range(self.config.grpo_epochs):
            for i in range(len(group_data['states'])):
                states = group_data['states'][i]
                actions = group_data['actions'][i]
                old_log_probs = group_data['log_probs_old'][i].detach()  # gradient 차단
                advantages = advantages_list[i]
                
                # Skip if empty
                if len(states) == 0:
                    continue
                
                try:
                    # 현재 log probs 계산 (gradient 필요)
                    current_log_probs = []
                    for j, state in enumerate(states):
                        # State가 gradient를 가지지 않도록 확인
                        state_clean = state.detach().requires_grad_(False)
                        
                        action_dist = self.get_action_distribution(state_clean)
                        log_prob = action_dist.log_prob(actions[j])
                        current_log_probs.append(log_prob)
                    
                    current_log_probs = torch.stack(current_log_probs)
                    
                    # NaN/Inf 체크
                    if torch.isnan(current_log_probs).any() or torch.isinf(current_log_probs).any():
                        logger.warning(f"⚠️ NaN/Inf in current_log_probs, skipping")
                        continue
                    
                    # 응답 마스크
                    response_mask = torch.ones_like(current_log_probs)
                    
                    # Policy loss 계산
                    policy_loss, metrics = compute_policy_loss(
                        old_log_probs, current_log_probs, advantages, response_mask,
                        self.config.clip_ratio_low, self.config.clip_ratio_high,
                        self.config.clip_ratio_dual, self.config.loss_avg_mode
                    )
                    
                    # Loss가 유효한지 체크
                    if torch.isnan(policy_loss) or torch.isinf(policy_loss):
                        logger.warning(f"⚠️ NaN/Inf in policy_loss, skipping")
                        continue
                    
                    # 역전파
                    self.optimizer.zero_grad()
                    policy_loss.backward()
                    
                    # 그래디언트 상태 체크
                    grad_norm = self._check_gradients()
                    
                    # 그래디언트 클리핑
                    torch.nn.utils.clip_grad_norm_(
                        list(self.qwen_model.model.parameters()) + list(self.action_head.parameters()),
                        max_norm=self.config.max_grad_norm
                    )
                    
                    self.optimizer.step()
                    
                    total_policy_loss += policy_loss.item()
                    for k, v in metrics.items():
                        total_metrics[k] += v
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error in policy update: {e}")
                    continue
        
        # 4. KL Controller 업데이트
        num_updates = len(group_data['states']) * self.config.grpo_epochs
        avg_kl = total_metrics.get('ppo_kl', 0.0) / max(num_updates, 1) if total_metrics else 0.0
        self.kl_controller.update(avg_kl, 1)
        
        # 5. 통계 저장
        avg_reward = np.mean(group_data['episode_rewards']) if group_data['episode_rewards'] else 0.0
        self.iteration_rewards.append(avg_reward)
        avg_policy_loss = total_policy_loss / max(num_updates, 1) if total_policy_loss > 0 else 0.0
        self.iteration_policy_losses.append(avg_policy_loss)
        self.iteration_kl_divs.append(avg_kl)
        
        result = {
            'avg_reward': avg_reward,
            'policy_loss': avg_policy_loss,
            'kl_div': avg_kl,
            'kl_coef': self.kl_controller.kl_coef,
            'num_episodes': len(group_data['episode_rewards'])
        }
        
        logger.info(f"✅ Iteration {self.current_iteration}: reward={avg_reward:.3f}, loss={avg_policy_loss:.3f}, kl={avg_kl:.3f}")
        
        return result

    def _check_gradients(self):
        """그래디언트 상태 체크"""
        total_norm = 0.0
        nan_count = 0
        
        for name, param in self.action_head.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    logger.warning(f"⚠️ NaN/Inf gradient in {name}")
                    param.grad.data.zero_()
                    nan_count += 1
        
        total_norm = total_norm ** (1. / 2)
        
        if nan_count > 0:
            logger.warning(f"⚠️ Zeroed {nan_count} NaN/Inf gradients")
        
        if total_norm > 100.0:
            logger.warning(f"⚠️ Large gradient norm: {total_norm:.2f}")
        
        return total_norm 