"""
GRPO Trainer for QWEN Model (Enhanced with EasyR1 Implementation)
================================================================

QWEN 모델을 GRPO 알고리즘으로 학습시키는 트레이너입니다.
EasyR1의 구현을 참조해서 개선된 GRPO 알고리즘을 적용합니다.

핵심 개선사항:
- 정확한 GRPO advantage 계산 (그룹 내 정규화)
- 적응형 KL Controller
- 이중 클리핑 policy loss
- 수치적 안정성 개선

기반 코드: EasyR1 verl/trainer/core_algos.py

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

# ===============================
# KL Controllers (from EasyR1)
# ===============================

class KLController(ABC):
    """KL coefficient controller base class"""
    kl_coef: float

    @abstractmethod
    def update(self, current_kl: float, n_steps: int):
        """Update kl_coef according to current KL."""
        ...

class AdaptiveKLController(KLController):
    """Adaptive KL controller from EasyR1"""

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
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
        self.kl_coef = init_kl_coef

    def update(self, current_kl: float, n_steps: int):
        pass

# ===============================
# Enhanced GRPO Config
# ===============================

@dataclass
class GRPOConfig:
    """Enhanced GRPO 학습 설정 (EasyR1 기반)"""
    # 학습 파라미터
    learning_rate: float = 1e-6  # 더 안정적인 학습률
    group_size: int = 4              # 그룹당 rollout 수
    num_iterations: int = 100        # 전체 학습 iteration
    grpo_epochs: int = 3             # 각 iteration당 최적화 epoch
    
    # GRPO 하이퍼파라미터
    gamma: float = 0.99              # 할인 팩터
    
    # KL 제어
    kl_type: str = "adaptive"        # "adaptive" or "fixed"
    kl_coef: float = 0.02            # 초기 KL 계수
    kl_target: float = 0.01          # 목표 KL (adaptive용)
    kl_horizon: float = 1000         # KL 적응 horizon
    kl_penalty: str = "kl"          # "kl", "abs", "mse", "low_var_kl", "full"
    
    # Policy 클리핑 (이중 클리핑)
    clip_ratio_low: float = 0.2      # 하한 클리핑
    clip_ratio_high: float = 0.2     # 상한 클리핑  
    clip_ratio_dual: float = 4.0     # 이중 클리핑
    
    # 기타
    entropy_coeff: float = 0.01      # 엔트로피 보너스
    max_grad_norm: float = 1.0       # 그래디언트 클리핑
    
    # 토큰 생성 파라미터
    max_new_tokens: int = 10         # placeholder에 추가할 최대 토큰 수
    temperature: float = 0.8         # 샘플링 온도
    
    # 시스템 설정
    device: str = "cuda"
    epsilon_std: float = 1e-6        # 수치 안정성
    loss_avg_mode: str = "token"     # "token" or "seq"
    
    # 저장 설정
    save_training_data: bool = True  # 학습 데이터 저장 여부
    save_dir: str = "training_results"  # 저장 디렉토리

# ===============================
# Utility Functions (from EasyR1)
# ===============================

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

# ===============================
# Enhanced Prompt Environment
# ===============================

class PromptEnvironment:
    """
    향상된 프롬프트 생성 환경 (EasyR1 기반)
    """
    
    def __init__(self, qwen_model, sd3_generator, clip_calculator, config: GRPOConfig):
        self.qwen_model = qwen_model
        self.sd3_generator = sd3_generator
        self.clip_calculator = clip_calculator
        self.config = config
        
        # 어휘 설정
        self.tokenizer = qwen_model.tokenizer
        self.vocab_size = len(self.tokenizer)
        
        # 유용한 토큰들을 필터링
        self.useful_token_ids = []
        for token_id in range(min(10000, self.vocab_size)):
            token_text = self.tokenizer.decode([token_id]).strip()
            
            if (len(token_text) >= 2 and  
                token_text.isalpha() and  
                not token_text.startswith('<') and  
                not token_text.startswith('[') and
                token_id not in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]):
                self.useful_token_ids.append(token_id)
        
        self.action_space_size = len(self.useful_token_ids)
        
        # 저장 관련 변수 초기화
        self.current_iteration = 0
        self.current_rollout = 0
        self.current_step = 0
        
        logger.info(f"🎮 Enhanced Environment initialized with {self.action_space_size} useful tokens")
    
    def set_iteration_info(self, iteration: int, rollout: int):
        """현재 iteration과 rollout 정보 설정"""
        self.current_iteration = iteration
        self.current_rollout = rollout
        self.current_step = 0
    
    def reset(self, user_prompt: str) -> torch.Tensor:
        """환경 리셋"""
        self.user_prompt = user_prompt
        base_placeholder = ", high quality, detailed"
        self.current_prompt = user_prompt + base_placeholder
        
        self.current_token_ids = self.tokenizer.encode(
            self.current_prompt, 
            add_special_tokens=False,
            return_tensors="pt"
        ).squeeze(0)
        
        self.current_step = 0
        state = self._get_state()
        return state
    
    def _get_state(self) -> torch.Tensor:
        """현재 상태 반환 (향상된 수치적 안정성)"""
        max_state_tokens = 20
        if len(self.current_token_ids) > max_state_tokens:
            state_token_ids = self.current_token_ids[-max_state_tokens:]
        else:
            state_token_ids = self.current_token_ids
        
        device = next(self.qwen_model.model.parameters()).device
        state_token_ids = state_token_ids.to(device)
        
        with torch.no_grad():
            embeddings = self.qwen_model.model.get_input_embeddings()(state_token_ids.unsqueeze(0))
            state = embeddings.mean(dim=1).squeeze(0)
            
            # 수치적 안정성을 위한 정규화
            state = F.normalize(state, dim=-1, eps=self.config.epsilon_std)
        
        return state
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        """액션 실행 (개선된 보상 계산)"""
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
    
    def _save_step_data(self, original_image, enhanced_image, reward: float):
        """스텝 데이터 저장"""
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
            
            with open(os.path.join(step_dir, "original_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(self.user_prompt)
            
            with open(os.path.join(step_dir, "enhanced_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(self.current_prompt)
            
            if original_image is not None:
                original_image.save(os.path.join(step_dir, "original_image.png"))
            
            if enhanced_image is not None:
                enhanced_image.save(os.path.join(step_dir, "enhanced_image.png"))
            
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
            
            logger.debug(f"💾 Saved step data to {step_dir}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save step data: {e}")
    
    def _calculate_reward(self) -> float:
        """CLIP을 사용한 보상 계산"""
        try:
            logger.debug(f"🖼️ Generating original image with: '{self.user_prompt}'")
            original_image = self.sd3_generator.generate_image(self.user_prompt)
            
            logger.debug(f"🖼️ Generating enhanced image with: '{self.current_prompt[:50]}...'")
            enhanced_image = self.sd3_generator.generate_image(self.current_prompt)
            
            if original_image is None and enhanced_image is None:
                return 0.0
            
            if enhanced_image is not None:
                reward = self.clip_calculator.calculate_reward(self.user_prompt, enhanced_image)
            else:
                reward = 0.0
            
            self._save_step_data(original_image, enhanced_image, reward)
            
            logger.debug(f"🎯 Reward: {reward:.3f} for '{self.user_prompt}' -> '{self.current_prompt[:50]}...'")
            return reward
            
        except Exception as e:
            logger.warning(f"⚠️ Reward calculation failed: {e}")
            return 0.0
    
    def get_action_space_size(self) -> int:
        return self.action_space_size
    
    def get_state_dimension(self) -> int:
        return self.qwen_model.model.config.hidden_size

# ===============================
# Enhanced GRPO Trainer
# ===============================

class GRPOTrainer:
    """
    향상된 GRPO 트레이너 (EasyR1 기반)
    """
    
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
        
        # 정책 네트워크 (수치적 안정성 개선)
        qwen_dtype = next(self.qwen_model.model.parameters()).dtype
        self.action_head = nn.Linear(
            self.env.get_state_dimension(),
            self.env.get_action_space_size(),
            dtype=qwen_dtype
        ).to(self.device)
        
        # 가중치 초기화
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        
        # 옵티마이저 (개선된 설정)
        self.optimizer = optim.AdamW(
            list(self.qwen_model.model.parameters()) + list(self.action_head.parameters()),
            lr=config.learning_rate,
            eps=1e-8,
            weight_decay=1e-6,
            betas=(0.9, 0.95)  # 더 안정적인 베타 값
        )
        
        # KL Controller 초기화
        self.kl_controller = self._get_kl_controller()
        
        # Reference model (frozen)
        self.ref_model = copy.deepcopy(self.qwen_model.model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # 학습 통계
        self.iteration_rewards = []
        self.iteration_policy_losses = []
        self.iteration_entropies = []
        self.iteration_kl_divs = []
        self.current_iteration = 0
        
        logger.info(f"🚀 Enhanced GRPO Trainer initialized with KL type: {config.kl_type}")
    
    def _get_kl_controller(self) -> KLController:
        """KL Controller 생성"""
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
        """상태에서 액션 확률 분포 계산 (수치적 안정성 개선)"""
        state = state.to(dtype=self.action_head.weight.dtype)
        action_logits = self.action_head(state)
        
        # NaN 및 무한값 처리
        if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
            logger.warning("⚠️ NaN or Inf detected in action logits, using uniform distribution")
            action_logits = torch.zeros_like(action_logits)
        
        # 로짓 클리핑 (더 강한 클리핑)
        action_logits = torch.clamp(action_logits, min=-20.0, max=20.0)
        
        return Categorical(logits=action_logits)
    
    @torch.no_grad()
    def compute_grpo_outcome_advantage(
        self, token_level_rewards: torch.Tensor, response_mask: torch.Tensor, 
        index: torch.Tensor, eps: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        EasyR1 스타일 GRPO advantage 계산
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
                # 단일 샘플인 경우 0으로 설정
                id2mean[idx] = scores[0].clone()  # 임시값
                id2std[idx] = torch.tensor(1.0, device=scores.device)

        for i in range(bsz):
            if len(id2score[index[i].item()]) > 1:
                scores[i] = (scores[i] - id2mean[index[i].item()]) / (id2std[index[i].item()] + eps)
            else:
                scores[i] = 0.0  # 단일 샘플은 advantage 0

        returns = scores.unsqueeze(-1) * response_mask
        return returns, returns
    
    def compute_kl(self, log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
        """KL divergence 계산 (EasyR1 스타일)"""
        log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
        
        if self.config.kl_penalty == "kl":
            return log_probs - ref_log_probs
        elif self.config.kl_penalty == "abs":
            return (log_probs - ref_log_probs).abs()
        elif self.config.kl_penalty == "mse":
            return 0.5 * (log_probs - ref_log_probs).square()
        elif self.config.kl_penalty == "low_var_kl":
            kl = (ref_log_probs - log_probs).clamp(-20.0, 20.0)
            kld = (kl.exp() - kl - 1).contiguous()
            return torch.clamp(kld, min=-10.0, max=10.0)
        elif self.config.kl_penalty == "full":
            return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)
        else:
            raise NotImplementedError(f"Unknown KL penalty: {self.config.kl_penalty}.")
    
    def compute_policy_loss(
        self, old_log_probs: torch.Tensor, log_probs: torch.Tensor,
        advantages: torch.Tensor, response_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        EasyR1 스타일 이중 클리핑 policy loss 계산
        """
        negative_approx_kl = log_probs - old_log_probs
        # KL 클리핑으로 수치적 안정성 확보
        negative_approx_kl = torch.clamp(negative_approx_kl, -20.0, 20.0)
        ratio = torch.exp(negative_approx_kl)
        
        # 클리핑된 ratio 계산
        clipped_ratio = torch.exp(
            torch.clamp(
                negative_approx_kl, 
                np.log(1.0 - self.config.clip_ratio_low), 
                np.log(1.0 + self.config.clip_ratio_high)
            )
        )

        # Metrics 계산
        metrics = {"ppo_kl": masked_mean(-negative_approx_kl, response_mask).item()}
        metrics["entropy_loss"] = average_loss(-log_probs, response_mask, mode=self.config.loss_avg_mode).item()

        # 3가지 loss 계산
        pg_loss = -advantages * ratio
        pg_loss2 = -advantages * clipped_ratio
        pg_loss3 = -advantages * self.config.clip_ratio_dual

        # 이중 클리핑 적용
        clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)
        metrics["pg_clipfrac_higher"] = masked_mean((pg_loss < pg_loss2).float(), response_mask).item()
        
        clipped_pg_loss_lower = torch.min(clipped_pg_loss_higher, pg_loss3)
        final_pg_loss = torch.where(advantages < 0, clipped_pg_loss_lower, clipped_pg_loss_higher)
        metrics["pg_clipfrac_lower"] = masked_mean(
            (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float(), 
            response_mask
        ).item()

        final_pg_loss = average_loss(final_pg_loss, response_mask, mode=self.config.loss_avg_mode)
        
        return final_pg_loss, metrics

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
            
            # 현재 iteration과 rollout 정보 설정
            self.env.set_iteration_info(getattr(self, 'current_iteration', 0), rollout_idx)
            
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