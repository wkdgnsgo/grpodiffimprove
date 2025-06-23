"""
GRPO (Group Relative Policy Optimization) Trainer
================================================

GRPO 알고리즘을 구현한 핵심 학습 모듈입니다.
VLM 프롬프트 개선을 위한 강화학습 트레이너입니다.

주요 기능:
1. GRPO 정책 업데이트
2. 그룹 기반 어드밴티지 계산
3. KL 발산 페널티
4. 참조 모델 관리

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

from typing import List, Dict, Tuple, Optional, Any
import logging
import numpy as np
import copy
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GRPOConfig:
    """GRPO 학습 설정 클래스"""
    learning_rate: float = 1e-5
    group_size: int = 4
    num_iterations: int = 20
    grpo_epochs: int = 2
    gamma: float = 0.99           # 할인 팩터
    kl_beta: float = 0.01         # KL 발산 페널티 계수
    clip_epsilon: float = 0.2     # 클리핑 범위
    entropy_coeff: float = 0.01   # 엔트로피 보너스 계수
    max_grad_norm: float = 1.0    # 그래디언트 클리핑
    epsilon_std: float = 1e-8     # 표준편차 정규화용 엡실론
    
    # 생성 파라미터
    max_new_tokens: int = 50
    temperature: float = 0.8
    
    # 디바이스 설정
    device: str = "auto"

class PolicyNetwork(nn.Module):
    """간단한 정책 네트워크 (VLM 프롬프트 개선용)"""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, vocab_size: int = 1000):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        """
        Args:
            x: 입력 상태 (프롬프트 임베딩)
        Returns:
            Categorical: 정책 분포
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return Categorical(logits=logits)

class GRPOTrainer:
    """
    Group Relative Policy Optimization (GRPO) 트레이너
    
    참조 코드 기반으로 완전히 재구현된 버전
    """
    
    def __init__(self, vlm_model, config: GRPOConfig):
        self.vlm = vlm_model
        self.config = config
        
        # 디바이스 설정
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        logger.info(f"🔧 GRPO Trainer initialized on device: {self.device}")
        
        # 정책 네트워크 초기화 (실제 학습 가능한 모델)
        self.policy_network = PolicyNetwork().to(self.device)
        
        # 옵티마이저 초기화
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=config.learning_rate)
        
        # 학습 통계
        self.training_stats = {
            'iteration': 0,
            'total_samples': 0,
            'avg_reward': 0.0,
            'policy_loss': 0.0,
            'entropy': 0.0,
            'kl_div': 0.0
        }
        
        logger.info("✅ GRPO Trainer ready for training")
    
    def collect_group_data(self, prompts: List[str]) -> Dict[str, Any]:
        """
        그룹 데이터 수집 (참조 코드의 rollout 단계와 유사)
        
        Args:
            prompts: 입력 프롬프트 리스트
            
        Returns:
            Dict: 수집된 그룹 데이터
        """
        logger.info(f"🔄 Collecting group data for {len(prompts)} prompts...")
        
        group_data = {
            'prompts': prompts,
            'enhanced_prompts': [],
            'states': [],
            'actions': [],
            'log_probs_old': [],  # 샘플링 시점의 로그 확률
            'rewards': [],
            'policy_distributions': []  # 실제 정책 분포 저장
        }
        
        # 정책 네트워크를 평가 모드로 설정 (rollout 단계)
        self.policy_network.eval()
        
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                try:
                    # 1. 프롬프트를 상태 벡터로 변환 (간단한 임베딩)
                    state = self._prompt_to_state(prompt)
                    
                    # 2. 정책 분포 계산
                    policy_dist = self.policy_network(state)
                    
                    # 3. 액션 샘플링 및 로그 확률 계산
                    action = policy_dist.sample()
                    log_prob_old = policy_dist.log_prob(action)
                    
                    # 4. 프롬프트 개선 (액션 기반)
                    enhanced_prompt = self._action_to_enhanced_prompt(prompt, action)
                    
                    # 5. 이미지 생성 및 보상 계산
                    image = self._generate_image(enhanced_prompt)
                    reward = self._calculate_reward(image, enhanced_prompt, prompt)
                    
                    # 6. 데이터 저장
                    group_data['enhanced_prompts'].append(enhanced_prompt)
                    group_data['states'].append(state)
                    group_data['actions'].append(action)
                    group_data['log_probs_old'].append(log_prob_old)
                    group_data['rewards'].append(reward)
                    group_data['policy_distributions'].append(policy_dist)
                    
                    logger.debug(f"  Prompt {i}: reward={reward:.4f}, log_prob={log_prob_old:.4f}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to process prompt {i}: {e}")
                    # 기본값으로 채우기
                    default_state = torch.zeros(128, device=self.device)
                    default_action = torch.tensor(0, device=self.device)
                    default_log_prob = torch.tensor(-2.0, device=self.device)
                    
                    group_data['enhanced_prompts'].append(f"enhanced: {prompt}")
                    group_data['states'].append(default_state)
                    group_data['actions'].append(default_action)
                    group_data['log_probs_old'].append(default_log_prob)
                    group_data['rewards'].append(0.0)
                    group_data['policy_distributions'].append(None)
        
        # 정책 네트워크를 다시 훈련 모드로 설정
        self.policy_network.train()
        
        logger.info(f"✅ Group data collected: {len(group_data['prompts'])} samples")
        logger.info(f"📊 Average reward: {np.mean(group_data['rewards']):.4f}")
        
        return group_data
    
    def _prompt_to_state(self, prompt: str) -> torch.Tensor:
        """프롬프트를 상태 벡터로 변환"""
        # 간단한 해시 기반 임베딩 (실제로는 더 정교한 방법 사용)
        hash_val = hash(prompt) % 1000000
        state = torch.zeros(128, device=self.device)
        state[:10] = torch.tensor([float(hash_val % (10**i)) / (10**i) for i in range(1, 11)], device=self.device)
        return state
    
    def _action_to_enhanced_prompt(self, prompt: str, action: torch.Tensor) -> str:
        """액션을 기반으로 프롬프트 개선"""
        action_val = action.item()
        enhancements = [
            "detailed, high quality",
            "artistic, beautiful",
            "photorealistic, 4k",
            "cinematic, dramatic lighting",
            "vibrant colors, sharp focus"
        ]
        enhancement = enhancements[action_val % len(enhancements)]
        return f"{prompt}, {enhancement}"
    
    def _generate_image(self, prompt: str):
        """이미지 생성 (플레이스홀더)"""
        return f"image_for_{prompt[:20]}"
    
    def _calculate_reward(self, image, enhanced_prompt: str, original_prompt: str) -> float:
        """보상 계산 (플레이스홀더)"""
        return min(len(enhanced_prompt) / 100.0, 1.0) + np.random.normal(0, 0.1)
    
    def _calculate_advantages_and_returns(self, group_data: Dict[str, Any]):
        """
        참조 코드 방식의 어드밴티지 계산
        
        1. 할인된 리턴 계산
        2. 그룹 정규화
        """
        logger.debug("🔄 Calculating advantages and returns (reference code style)...")
        
        rewards = group_data['rewards']
        group_size = len(rewards)
        
        # 1. 할인된 리턴 계산 (각 샘플은 단일 스텝이므로 단순화)
        returns = np.array(rewards, dtype=np.float32)
        
        # 2. 그룹 평균 기반 어드밴티지 계산
        if group_size > 1:
            group_mean = np.mean(returns)
            group_std = np.std(returns)
            
            # 정규화: (x - mean) / (std + epsilon)
            advantages = (returns - group_mean) / (group_std + self.config.epsilon_std)
        else:
            advantages = np.array([0.0])
        
        # 3. 텐서로 변환
        group_data['returns'] = [torch.tensor(ret, dtype=torch.float32, device=self.device) for ret in returns]
        group_data['advantages'] = [torch.tensor(adv, dtype=torch.float32, device=self.device) for adv in advantages]
        
        logger.debug(f"📊 Returns: mean={np.mean(returns):.4f}, std={np.std(returns):.4f}")
        logger.debug(f"📊 Advantages: mean={np.mean(advantages):.4f}, std={np.std(advantages):.4f}")
    
    def grpo_update(self, group_data: Dict[str, Any]) -> Dict[str, float]:
        """
        GRPO 업데이트 (참조 코드 방식)
        
        Args:
            group_data: 수집된 그룹 데이터
            
        Returns:
            Dict: 업데이트 메트릭
        """
        logger.info("🔄 Starting GRPO update...")
        
        # 1. 어드밴티지 계산
        self._calculate_advantages_and_returns(group_data)
        
        # 2. 참조 모델 생성 (현재 모델의 완전한 복사본)
        policy_ref = PolicyNetwork().to(self.device)
        policy_ref.load_state_dict(self.policy_network.state_dict())
        policy_ref.eval()
        
        # 3. 여러 에포크 업데이트
        total_policy_loss = 0.0
        total_entropy = 0.0
        total_kl_div = 0.0
        
        for epoch in range(self.config.grpo_epochs):
            metrics = self._grpo_epoch_update(group_data, policy_ref)
            total_policy_loss += metrics['policy_loss']
            total_entropy += metrics['entropy']
            total_kl_div += metrics['kl_div']
        
        # 4. 평균 메트릭 계산
        avg_metrics = {
            'policy_loss': total_policy_loss / self.config.grpo_epochs,
            'entropy': total_entropy / self.config.grpo_epochs,
            'kl_div': total_kl_div / self.config.grpo_epochs,
            'avg_reward': np.mean(group_data['rewards'])
        }
        
        # 5. 통계 업데이트
        self.training_stats.update(avg_metrics)
        self.training_stats['iteration'] += 1
        self.training_stats['total_samples'] += len(group_data['prompts'])
        
        logger.info(f"✅ GRPO update complete:")
        logger.info(f"  Policy Loss: {avg_metrics['policy_loss']:.4f}")
        logger.info(f"  Entropy: {avg_metrics['entropy']:.4f}")
        logger.info(f"  KL Div: {avg_metrics['kl_div']:.4f}")
        logger.info(f"  Avg Reward: {avg_metrics['avg_reward']:.4f}")
        
        return avg_metrics
    
    def _grpo_epoch_update(self, group_data: Dict[str, Any], policy_ref: PolicyNetwork) -> Dict[str, float]:
        """
        단일 에포크 GRPO 업데이트 (참조 코드 방식)
        """
        self.optimizer.zero_grad()
        
        total_policy_loss = 0.0
        total_entropy = 0.0
        total_kl_div = 0.0
        batch_size = len(group_data['prompts'])
        
        for i in range(batch_size):
            # 1. 현재 정책으로 재계산
            state = group_data['states'][i]
            action = group_data['actions'][i]
            old_log_prob = group_data['log_probs_old'][i]
            advantage = group_data['advantages'][i]
            
            # 2. 현재 정책 분포
            current_policy_dist = self.policy_network(state)
            current_log_prob = current_policy_dist.log_prob(action)
            
            # 3. 참조 정책 분포
            with torch.no_grad():
                ref_policy_dist = policy_ref(state)
                ref_log_prob = ref_policy_dist.log_prob(action)
            
            # 4. 정책 비율 계산
            log_ratio = current_log_prob - old_log_prob.detach()
            ratio = torch.exp(log_ratio)
            
            # 5. 클리핑된 서로게이트 손실 (PPO/GRPO)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantage
            policy_loss_i = -torch.min(surr1, surr2)  # 음수: 최대화 -> 최소화
            
            # 6. 엔트로피 계산 (실제 정책 분포에서)
            entropy_i = current_policy_dist.entropy()
            
            # 7. KL divergence 계산
            kl_div_i = torch.distributions.kl_divergence(ref_policy_dist, current_policy_dist)
            
            total_policy_loss += policy_loss_i
            total_entropy += entropy_i
            total_kl_div += kl_div_i
        
        # 8. 평균 계산
        avg_policy_loss = total_policy_loss / batch_size
        avg_entropy = total_entropy / batch_size
        avg_kl_div = total_kl_div / batch_size
        
        # 9. 총 손실: 정책 손실 + KL 페널티 - 엔트로피 보너스
        total_loss = avg_policy_loss + self.config.kl_beta * avg_kl_div - self.config.entropy_coeff * avg_entropy
        
        # 10. 역전파
        total_loss.backward()
        
        # 11. 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.config.max_grad_norm)
        
        # 12. 옵티마이저 스텝
        self.optimizer.step()
        
        return {
            'policy_loss': float(avg_policy_loss.detach()),
            'entropy': float(avg_entropy.detach()),
            'kl_div': float(avg_kl_div.detach()),
            'total_loss': float(total_loss.detach())
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """현재 학습 통계 반환"""
        return self.training_stats.copy()
    
    def save_checkpoint(self, checkpoint_path: str):
        """체크포인트 저장"""
        try:
            checkpoint = {
                'policy_network_state_dict': self.policy_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'training_stats': self.training_stats
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_stats = checkpoint['training_stats']
            logger.info(f"📥 Checkpoint loaded: {checkpoint_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")


if __name__ == "__main__":
    # GRPO Trainer 테스트 코드
    print("🧪 GRPO Trainer Test (Reference Code Style)")
    print("=" * 50)
    
    try:
        # Mock VLM 모델
        class MockVLM:
            def enhance_prompt(self, prompt):
                return f"enhanced: {prompt}"
        
        # 설정 및 트레이너 초기화
        config = GRPOConfig(
            learning_rate=1e-4,
            group_size=3,
            num_iterations=2,
            grpo_epochs=2
        )
        
        mock_vlm = MockVLM()
        trainer = GRPOTrainer(mock_vlm, config)
        
        print("✅ GRPO Trainer initialized successfully")
        print(f"📊 Training stats: {trainer.get_training_stats()}")
        
        # 테스트 프롬프트
        test_prompts = ["a cat", "sunset", "mountain"]
        
        print("\n🔄 Testing group data collection:")
        group_data = trainer.collect_group_data(test_prompts)
        print(f"  Collected data for {len(group_data['prompts'])} prompts")
        print(f"  Average reward: {np.mean(group_data['rewards']):.4f}")
        
        print("\n🔄 Testing GRPO update:")
        metrics = trainer.grpo_update(group_data)
        print(f"  Update metrics: {metrics}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nUsage:")
    print("from training.grpo_trainer import GRPOTrainer, GRPOConfig")
    print("config = GRPOConfig()")
    print("trainer = GRPOTrainer(vlm_model, config)")
    print("group_data = trainer.collect_group_data(prompts)")
    print("metrics = trainer.grpo_update(group_data)") 