"""
GRPO (Group Relative Policy Optimization) Trainer
================================================

GRPO 알고리즘을 구현한 핵심 학습 모듈입니다.
Qwen2.5-VL 자체를 정책 네트워크로 사용하여 프롬프트 개선을 학습합니다.

주요 기능:
1. Qwen2.5-VL을 직접 정책 네트워크로 사용
2. GRPO 정책 업데이트
3. 그룹 기반 어드밴티지 계산
4. KL 발산 페널티
5. 참조 모델 관리

GRPO vs PPO 차이점:
- PPO: 개별 샘플 기반 어드밴티지
- GRPO: 그룹 내 상대적 어드밴티지 (더 안정적)

Author: AI Assistant
Date: 2025-01-22
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import logging
import copy

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GRPOConfig:
    """GRPO 학습 설정 클래스"""
    learning_rate: float = 5e-6
    group_size: int = 4
    num_iterations: int = 100
    grpo_epochs: int = 3
    gamma: float = 0.99           # 할인 팩터
    kl_beta: float = 0.02         # KL 발산 페널티 계수
    clip_epsilon: float = 0.2     # 클리핑 범위
    entropy_coeff: float = 0.01   # 엔트로피 보너스 계수
    max_grad_norm: float = 1.0    # 그래디언트 클리핑
    epsilon_std: float = 1e-8     # 표준편차 정규화용 엡실론
    
    # 토큰 생성 파라미터 (config에서 로드됨)
    max_new_tokens: int = 25      # 최대 생성 토큰 수
    max_prompt_length: int = 77   # 최대 프롬프트 길이 (CLIP 제한)
    max_sequence_length: int = 102 # 최대 시퀀스 길이 (prompt + new_tokens)
    vocab_size: int = 50000       # 어휘 크기 (모델에서 자동 설정)
    temperature: float = 0.8
    
    # 디바이스 설정
    device: str = "auto"

class GRPOTrainer:
    """Qwen2.5-VL을 직접 정책 네트워크로 사용하는 GRPO 트레이너"""
    
    def __init__(self, vlm_model, sd_generator, clip_reward, config: GRPOConfig):
        """
        GRPO Trainer 초기화
        
        Args:
            vlm_model: VLMWrapper 인스턴스 (Qwen2.5-VL 포함)
            sd_generator: SD3 생성기 (동결됨)
            clip_reward: CLIP 보상 계산기 (동결됨)
            config: GRPO 설정
        """
        # --- Core Components ---
        self.vlm_policy = vlm_model  # VLM이 곧 정책 네트워크
        self.sd_generator = sd_generator  # 동결된 SD3 파이프라인
        self.clip_reward = clip_reward    # 동결된 CLIP 보상 모델
        self.config = config
        
        # --- Device Setup ---
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        # --- VLM Policy Network Validation ---
        if not hasattr(self.vlm_policy, 'model') or self.vlm_policy.model is None:
            raise ValueError("VLM model is not loaded. Please ensure VLMWrapper has loaded Qwen2.5-VL model.")
        
        # --- Tokenizer Setup ---
        self.tokenizer = self.vlm_policy.tokenizer
        self.vocab_size = len(self.tokenizer)
        
        # --- Reference Policy Network (Copy of current VLM) ---
        self.vlm_policy_ref = None
        self._create_reference_policy()
        
        # --- Optimizer for VLM Policy ---
        self.vlm_optimizer = optim.Adam(self.vlm_policy.model.parameters(), lr=config.learning_rate)
        
        # --- Training Statistics ---
        self.training_stats = {
            'iteration': 0,
            'total_samples': 0,
            'avg_reward': 0.0,
            'policy_loss': 0.0,
            'entropy': 0.0,
            'kl_div': 0.0
        }
        
        # --- Lists for Logging/Plotting ---
        self.iteration_rewards = []
        self.iteration_policy_losses = []
        self.iteration_entropies = []
        self.iteration_kl_divs = []
        
        logger.info(f"🚀 GRPO Trainer initialized with device: {self.device}")
        logger.info(f"📊 VLM policy parameters: {sum(p.numel() for p in self.vlm_policy.model.parameters()):,}")
        logger.info(f"📝 Vocab size: {self.vocab_size}")
    
    def _create_reference_policy(self):
        """참조 정책 생성 (현재 VLM의 deepcopy)"""
        try:
            logger.info("📋 Creating reference policy from current VLM...")
            
            # 참조 정책은 VLM 모델의 완전한 복사본
            self.vlm_policy_ref = copy.deepcopy(self.vlm_policy.model)
            self.vlm_policy_ref.eval()
            
            # 참조 정책의 그래디언트 비활성화
            for param in self.vlm_policy_ref.parameters():
                param.requires_grad = False
            
            logger.info("✅ Reference policy created and frozen")
            
        except Exception as e:
            logger.error(f"❌ Failed to create reference policy: {e}")
            self.vlm_policy_ref = None
    
    def collect_group_trajectories(self, prompts: List[str]) -> Dict[str, Any]:
        """
        그룹 궤적 수집 - VLM 정책을 사용한 토큰별 순차 생성
        
        각 프롬프트에 대해:
        1. VLM으로 토큰별 순차 생성 (rollout)
        2. 각 스텝에서 state = user_prompt + 지금까지_생성된_토큰들
        3. action = 다음_토큰_선택
        4. 완성된 프롬프트로 이미지 생성 및 보상 계산
        """
        logger.info(f"🔄 Collecting group trajectories for {len(prompts)} prompts...")
        
        # --- Group Data Storage ---
        group_states_list: List[torch.Tensor] = []
        group_actions_list: List[torch.Tensor] = []
        group_log_probs_old_list: List[torch.Tensor] = []
        group_rewards_list: List[List[float]] = []
        
        episode_rewards_in_iter = []
        episode_lengths_in_iter = []
        
        # --- Set VLM Policy to Evaluation Mode for Rollout ---
        self.vlm_policy.model.eval()
        
        for rollout_idx, prompt in enumerate(prompts):
            logger.debug(f"📝 Processing rollout {rollout_idx+1}/{len(prompts)}: '{prompt[:50]}...'")
            
            # --- Single Rollout Data ---
            rollout_states: List[torch.Tensor] = []
            rollout_actions: List[torch.Tensor] = []
            rollout_log_probs: List[torch.Tensor] = []
            rollout_rewards: List[float] = []
            
            try:
                # --- VLM Sequential Generation (Token-by-token) ---
                generation_result = self.vlm_policy.generate_sequence(
                    prompt=prompt,
                    max_new_tokens=self.config.max_new_tokens
                )
                
                generated_text = generation_result['generated_text']
                states = generation_result['states']  # List of state tensors
                actions = generation_result['actions']  # List of action tensors
                log_probs = generation_result['log_probs']  # List of log_prob tensors
                
                # --- Environment Interaction: Text→Image→Reward ---
                try:
                    # SD3로 이미지 생성
                    generated_image = self.sd_generator.generate_image(generated_text)
                    
                    # CLIP으로 보상 계산
                    final_reward = self.clip_reward.calculate_reward(
                        image=generated_image,
                        text=generated_text
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Image generation/reward failed: {e}")
                    final_reward = 0.0
                
                # --- Store Rollout Data ---
                if states and actions and log_probs:
                    # 각 스텝에 동일한 최종 보상 할당 (할인 적용 예정)
                    step_rewards = [final_reward] * len(states)
                    
                    rollout_states = states
                    rollout_actions = actions
                    rollout_log_probs = log_probs
                    rollout_rewards = step_rewards
                    
                    episode_rewards_in_iter.append(final_reward)
                    episode_lengths_in_iter.append(len(states))
                    
                    logger.debug(f"✅ Generated: '{generated_text[:50]}...', Reward: {final_reward:.4f}, Steps: {len(states)}")
                else:
                    logger.warning(f"⚠️ Empty generation result for prompt: {prompt}")
                    episode_rewards_in_iter.append(0.0)
                    episode_lengths_in_iter.append(0)
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to process prompt '{prompt}': {e}")
                episode_rewards_in_iter.append(0.0)
                episode_lengths_in_iter.append(0)
            
            # --- Store Completed Rollout Data as Tensors ---
            if rollout_states:
                # Handle variable length sequences by padding or truncating
                try:
                    # Try to stack if all have same length
                    if len(set(s.shape for s in rollout_states)) == 1:
                        group_states_list.append(torch.stack(rollout_states))
                        group_actions_list.append(torch.stack(rollout_actions).squeeze())
                        group_log_probs_old_list.append(torch.stack(rollout_log_probs).squeeze())
                    else:
                        # Pad to same length
                        max_len = max(s.shape[-1] for s in rollout_states)
                        padded_states = []
                        for state in rollout_states:
                            if state.shape[-1] < max_len:
                                pad_size = max_len - state.shape[-1]
                                padded_state = torch.cat([state, torch.zeros(pad_size, dtype=state.dtype, device=state.device)])
                            else:
                                padded_state = state
                            padded_states.append(padded_state)
                        
                        group_states_list.append(torch.stack(padded_states))
                        group_actions_list.append(torch.stack(rollout_actions).squeeze())
                        group_log_probs_old_list.append(torch.stack(rollout_log_probs).squeeze())
                    
                    group_rewards_list.append(rollout_rewards)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to stack rollout data: {e}, using individual tensors")
                    # Fallback: store as individual tensors
                    group_states_list.append(rollout_states)
                    group_actions_list.append(rollout_actions)
                    group_log_probs_old_list.append(rollout_log_probs)
                    group_rewards_list.append(rollout_rewards)
            else:
                # Empty rollout placeholder
                group_states_list.append([])
                group_actions_list.append([])
                group_log_probs_old_list.append([])
                group_rewards_list.append([])
        
        # --- Set VLM Policy back to Training Mode ---
        self.vlm_policy.model.train()
        
        logger.info(f"📊 Collected {len(group_states_list)} trajectories")
        avg_reward = np.mean(episode_rewards_in_iter) if episode_rewards_in_iter else 0.0
        avg_length = np.mean(episode_lengths_in_iter) if episode_lengths_in_iter else 0.0
        logger.info(f"📊 Average reward: {avg_reward:.4f}, Average length: {avg_length:.1f}")
        
        return {
            'group_states_list': group_states_list,
            'group_actions_list': group_actions_list,
            'group_log_probs_old_list': group_log_probs_old_list,
            'group_rewards_list': group_rewards_list,
            'episode_rewards': episode_rewards_in_iter,
            'episode_lengths': episode_lengths_in_iter
        }
    
    def calculate_group_advantages(self, group_data: Dict[str, Any]) -> List[torch.Tensor]:
        """
        그룹 상대적 어드밴티지 계산 (할인된 리턴 방법)
        """
        logger.debug("🔄 Calculating group relative advantages...")
        
        group_rewards_list = group_data['group_rewards_list']
        
        # --- Lists for Group Advantage Calculation ---
        group_advantages_list: List[torch.Tensor] = []
        all_raw_advantages_in_group: List[float] = []
        temp_raw_advantages_tensors: List[torch.Tensor] = []
        
        # --- First Pass: Calculate RAW Discounted Returns-to-go ---
        for i, rollout_rewards in enumerate(group_rewards_list):
            rollout_len = len(rollout_rewards)
            rollout_raw_advantages = torch.zeros(rollout_len, dtype=torch.float32, device=self.device)
            
            if rollout_len > 0:
                # Calculate discounted returns (G_t = r_t + gamma*G_{t+1})
                discounted_return = 0.0
                for t in reversed(range(rollout_len)):
                    discounted_return = rollout_rewards[t] + self.config.gamma * discounted_return
                    rollout_raw_advantages[t] = discounted_return
                
                # Store raw advantages for normalization
                temp_raw_advantages_tensors.append(rollout_raw_advantages)
                all_raw_advantages_in_group.extend(rollout_raw_advantages.cpu().numpy())
            else:
                temp_raw_advantages_tensors.append(torch.empty((0,), device=self.device))
        
        # --- Calculate Mean/Std of ALL RAW Discounted Returns ---
        if len(all_raw_advantages_in_group) > 1:
            group_mean_advantage = np.mean(all_raw_advantages_in_group)
            group_std_advantage = np.std(all_raw_advantages_in_group)
        elif len(all_raw_advantages_in_group) == 1:
            group_mean_advantage = all_raw_advantages_in_group[0]
            group_std_advantage = 0.0
        else:
            group_mean_advantage = 0.0
            group_std_advantage = 0.0
            logger.warning("⚠️ No advantages calculated in group (all rollouts empty?)")
        
        # --- Second Pass: Normalize Raw Discounted Returns ---
        for i, raw_advantages_tensor in enumerate(temp_raw_advantages_tensors):
            if raw_advantages_tensor.nelement() > 0:
                # Normalize using group's mean/std
                normalized_advantages = (raw_advantages_tensor - group_mean_advantage) / (group_std_advantage + self.config.epsilon_std)
            else:
                normalized_advantages = raw_advantages_tensor
            
            group_advantages_list.append(normalized_advantages)
        
        logger.debug(f"📊 Group advantages: mean={group_mean_advantage:.4f}, std={group_std_advantage:.4f}")
        
        return group_advantages_list
    
    def update_grpo_policy(self, group_data: Dict[str, Any], group_advantages_list: List[torch.Tensor]) -> Dict[str, float]:
        """
        GRPO 정책 업데이트 수행
        """
        logger.info("🔄 Performing GRPO policy update...")
        
        # --- Create Reference Policy (Copy current VLM state) ---
        if self.vlm_policy_ref is None:
            self._create_reference_policy()
        else:
            # Update reference policy to current state
            self.vlm_policy_ref.load_state_dict(self.vlm_policy.model.state_dict())
            self.vlm_policy_ref.eval()
        
        # --- Extract Group Data ---
        group_states_list = group_data['group_states_list']
        group_actions_list = group_data['group_actions_list']
        group_log_probs_old_list = group_data['group_log_probs_old_list']
        
        # --- GRPO Update Loop ---
        total_policy_loss = 0.0
        total_kl_div = 0.0
        total_entropy = 0.0
        update_count = 0
        
        for epoch in range(self.config.grpo_epochs):
            epoch_policy_loss = 0.0
            epoch_kl_div = 0.0
            epoch_entropy = 0.0
            
            # --- Process Each Trajectory in Group ---
            for i in range(len(group_states_list)):
                states = group_states_list[i]
                actions = group_actions_list[i]
                log_probs_old = group_log_probs_old_list[i]
                advantages = group_advantages_list[i]
                
                # Handle different data types (tensor vs list)
                if isinstance(states, list):
                    if not states:  # Empty list
                        continue
                    # Convert list to tensor if possible
                    try:
                        if len(set(s.shape for s in states)) == 1:
                            states = torch.stack(states)
                            actions = torch.stack(actions).squeeze() if isinstance(actions, list) else actions
                            log_probs_old = torch.stack(log_probs_old).squeeze() if isinstance(log_probs_old, list) else log_probs_old
                        else:
                            logger.warning(f"⚠️ Skipping trajectory {i} due to inconsistent tensor shapes")
                            continue
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to process trajectory {i}: {e}")
                        continue
                elif hasattr(states, 'nelement') and states.nelement() == 0:  # Empty tensor
                    continue
                
                # --- Calculate Current Policy Log Probs ---
                try:
                    current_log_probs = []
                    current_entropies = []
                    ref_log_probs = []
                    
                    for step in range(len(states)):
                        state = states[step].unsqueeze(0)  # Add batch dimension
                        action = actions[step]
                        
                        # Current policy
                        current_policy_dist = self.vlm_policy.forward(state)
                        current_log_prob = self.vlm_policy.get_log_prob(state, action)
                        current_entropy = current_policy_dist.entropy()
                        
                        # Reference policy
                        with torch.no_grad():
                            ref_outputs = self.vlm_policy_ref(input_ids=state, return_dict=True)
                            ref_logits = ref_outputs.logits[:, -1, :]
                            ref_policy_dist = Categorical(logits=ref_logits)
                            ref_log_prob = ref_policy_dist.log_prob(action)
                        
                        current_log_probs.append(current_log_prob)
                        current_entropies.append(current_entropy)
                        ref_log_probs.append(ref_log_prob)
                    
                    if not current_log_probs:
                        continue
                    
                    current_log_probs = torch.stack(current_log_probs)
                    current_entropies = torch.stack(current_entropies)
                    ref_log_probs = torch.stack(ref_log_probs)
                    
                    # Ensure log_probs_old is tensor
                    if isinstance(log_probs_old, list):
                        log_probs_old = torch.stack(log_probs_old).squeeze()
                    
                    # --- GRPO Loss Calculation ---
                    log_ratio = current_log_probs - log_probs_old.detach()
                    ratio = torch.exp(log_ratio)
                    
                    # PPO-style clipped surrogate loss
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # KL divergence penalty
                    log_ratio_ref_curr = ref_log_probs - current_log_probs.detach()
                    kl_div = torch.exp(log_ratio_ref_curr) - log_ratio_ref_curr - 1
                    kl_div = torch.relu(kl_div).mean()
                    
                    # Entropy bonus
                    entropy = current_entropies.mean()
                    
                    # Combined loss
                    total_loss = policy_loss + self.config.kl_beta * kl_div - self.config.entropy_coeff * entropy
                    
                    # --- Backward Pass ---
                    self.vlm_optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.vlm_policy.model.parameters(), self.config.max_grad_norm)
                    self.vlm_optimizer.step()
                    
                    # --- Accumulate Metrics ---
                    epoch_policy_loss += policy_loss.item()
                    epoch_kl_div += kl_div.item()
                    epoch_entropy += entropy.item()
                    update_count += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to update trajectory {i}: {e}")
                    continue
            
            total_policy_loss += epoch_policy_loss
            total_kl_div += epoch_kl_div
            total_entropy += epoch_entropy
        
        # --- Calculate Average Metrics ---
        if update_count > 0:
            avg_policy_loss = total_policy_loss / update_count
            avg_kl_div = total_kl_div / update_count
            avg_entropy = total_entropy / update_count
        else:
            avg_policy_loss = 0.0
            avg_kl_div = 0.0
            avg_entropy = 0.0
        
        logger.info(f"✅ GRPO update complete - Policy Loss: {avg_policy_loss:.4f}, KL: {avg_kl_div:.4f}, Entropy: {avg_entropy:.4f}")
        
        return {
            'policy_loss': avg_policy_loss,
            'kl_div': avg_kl_div,
            'entropy': avg_entropy
        }
    
    def train_iteration(self, prompts: List[str]) -> Dict[str, float]:
        """
        단일 GRPO 학습 반복 수행
        """
        # --- 1. Collect Group of Trajectories (Rollout Phase) ---
        group_data = self.collect_group_trajectories(prompts)
        
        # --- 2. Calculate Group Relative Advantages ---
        group_advantages_list = self.calculate_group_advantages(group_data)
        
        # --- 3. Perform GRPO Update ---
        update_metrics = self.update_grpo_policy(group_data, group_advantages_list)
        
        # --- 4. Update Training Statistics ---
        avg_reward = np.mean(group_data['episode_rewards']) if group_data['episode_rewards'] else 0.0
        
        self.training_stats['iteration'] += 1
        self.training_stats['avg_reward'] = avg_reward
        self.training_stats['policy_loss'] = update_metrics['policy_loss']
        self.training_stats['entropy'] = update_metrics['entropy']
        self.training_stats['kl_div'] = update_metrics['kl_div']
        
        # --- 5. Store for Logging/Plotting ---
        self.iteration_rewards.append(avg_reward)
        self.iteration_policy_losses.append(update_metrics['policy_loss'])
        self.iteration_entropies.append(update_metrics['entropy'])
        self.iteration_kl_divs.append(update_metrics['kl_div'])
        
        return {
            'avg_reward': avg_reward,
            'policy_loss': update_metrics['policy_loss'],
            'entropy': update_metrics['entropy'],
            'kl_div': update_metrics['kl_div']
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """현재 학습 통계 반환"""
        return self.training_stats.copy()
    
    def save_checkpoint(self, checkpoint_path: str):
        """체크포인트 저장 - VLM 모델 상태 저장"""
        try:
            checkpoint = {
                'vlm_model_state_dict': self.vlm_policy.model.state_dict(),
                'optimizer_state_dict': self.vlm_optimizer.state_dict(),
                'tokenizer': self.tokenizer,
                'config': self.config,
                'training_stats': self.training_stats,
                'iteration_rewards': self.iteration_rewards,
                'iteration_policy_losses': self.iteration_policy_losses,
                'iteration_entropies': self.iteration_entropies,
                'iteration_kl_divs': self.iteration_kl_divs
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드 - VLM 모델 상태 로드"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.vlm_policy.model.load_state_dict(checkpoint['vlm_model_state_dict'])
            self.vlm_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_stats = checkpoint['training_stats']
            
            # Load logging data if available
            if 'iteration_rewards' in checkpoint:
                self.iteration_rewards = checkpoint['iteration_rewards']
                self.iteration_policy_losses = checkpoint['iteration_policy_losses']
                self.iteration_entropies = checkpoint['iteration_entropies']
                self.iteration_kl_divs = checkpoint['iteration_kl_divs']
            
            # 참조 정책 재생성
            self._create_reference_policy()
            
            logger.info(f"📂 Checkpoint loaded: {checkpoint_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")

# 테스트용 Mock 클래스들
if __name__ == "__main__":
    class MockVLM:
        def __init__(self):
            self.model = torch.nn.Linear(10, 10)  # 더미 모델
            self.tokenizer = None
            
        def forward(self, input_ids):
            return Categorical(logits=torch.randn(1, 1000))
            
        def generate_sequence(self, prompt, max_new_tokens=10):
            return {
                'generated_text': f"enhanced {prompt}",
                'generated_ids': torch.randint(0, 1000, (1, 10)),
                'states': [torch.randint(0, 1000, (5,)) for _ in range(3)],
                'actions': [torch.randint(0, 1000, (1,)) for _ in range(3)],
                'log_probs': [torch.randn(1) for _ in range(3)]
            }
            
        def get_log_prob(self, input_ids, target_token):
            return torch.randn(1)
    
    class MockSDGenerator:
        def generate_image(self, prompt):
            return f"image_for_{prompt[:20]}"
    
    class MockCLIPReward:
        def calculate_reward(self, image, text):
            return np.random.uniform(0.3, 0.8)
    
    # 테스트
    config = GRPOConfig()
    trainer = GRPOTrainer(
        vlm_model=MockVLM(),
        sd_generator=MockSDGenerator(),
        clip_reward=MockCLIPReward(),
        config=config
    )
    
    test_prompts = ["a beautiful sunset", "a cat in the garden"]
    metrics = trainer.train_iteration(test_prompts)
    print(f"Test metrics: {metrics}") 