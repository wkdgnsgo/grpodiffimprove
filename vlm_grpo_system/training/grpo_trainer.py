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
    
    # 생성 파라미터
    max_new_tokens: int = 50
    temperature: float = 0.8
    
    # 디바이스 설정
    device: str = "auto"

class GRPOTrainer:
    """
    GRPO 알고리즘을 사용한 VLM 학습 클래스
    
    이 클래스는 GRPO 논문의 핵심 아이디어를 구현합니다:
    1. 그룹 내에서 상대적 성능 비교
    2. 참조 모델과의 KL 발산 제한
    3. 안정적인 정책 업데이트
    
    Attributes:
        config (GRPOConfig): 학습 설정
        vlm: VLM 모델 (학습 대상)
        vlm_ref: 참조 VLM 모델 (고정)
        optimizer: 옵티마이저
        device: 연산 디바이스
    """
    
    def __init__(self, 
                 vlm_model,
                 config: GRPOConfig):
        """
        GRPO Trainer 초기화
        
        Args:
            vlm_model: 학습할 VLM 모델
            config (GRPOConfig): GRPO 학습 설정
        """
        self.config = config
        self.vlm = vlm_model
        
        # 디바이스 설정
        if config.device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("🍎 Using Apple Silicon MPS for GRPO")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("🚀 Using CUDA GPU for GRPO")
            else:
                self.device = torch.device("cpu")
                logger.info("💻 Using CPU for GRPO")
        else:
            self.device = torch.device(config.device)
        
        # VLM을 디바이스로 이동
        self.vlm = self.vlm.to(self.device)
        
        # 참조 모델 생성 (매 iteration마다 업데이트)
        self.vlm_ref = None
        
        # 옵티마이저 설정
        self.optimizer = optim.AdamW(
            self.vlm.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # 학습 통계
        self.training_stats = {
            'iteration': 0,
            'total_samples': 0,
            'avg_reward': 0.0,
            'policy_loss': 0.0,
            'kl_divergence': 0.0,
            'entropy': 0.0
        }
        
        logger.info(f"🔧 GRPO Trainer initialized with config: {config}")
    
    def collect_group_data(self, prompts: List[str]) -> Dict[str, Any]:
        """
        그룹 데이터 수집: 프롬프트 개선 및 보상 계산
        
        이 메서드는 GRPO의 핵심 데이터 수집 단계입니다:
        1. 각 프롬프트를 VLM으로 개선
        2. 개선된 프롬프트로 이미지 생성
        3. CLIP으로 보상 계산
        4. 로그 확률 계산
        
        Args:
            prompts (List[str]): 입력 프롬프트 그룹
            
        Returns:
            Dict[str, Any]: 수집된 그룹 데이터
        """
        logger.debug(f"📊 Collecting group data for {len(prompts)} prompts")
        
        group_data = {
            'prompts': prompts,
            'enhanced_prompts': [],
            'images': [],
            'rewards': [],
            'log_probs': [],
            'ref_log_probs': [],
            'advantages': [],
            'returns': []
        }
        
        # 참조 모델 업데이트 (현재 정책의 복사본)
        self._update_reference_model()
        
        # 각 프롬프트에 대해 데이터 수집
        for prompt in prompts:
            try:
                # 1. VLM으로 프롬프트 개선
                enhanced_prompt, log_prob = self._enhance_prompt_with_logprob(prompt)
                
                # 2. 참조 모델로 로그 확률 계산
                ref_log_prob = self._calculate_reference_logprob(prompt, enhanced_prompt)
                
                # 3. 이미지 생성 (실제 구현에서는 SD3 사용)
                image = self._generate_image(enhanced_prompt)
                
                # 4. 보상 계산 (실제 구현에서는 CLIP 사용)
                reward = self._calculate_reward(image, enhanced_prompt, prompt)
                
                # 데이터 저장
                group_data['enhanced_prompts'].append(enhanced_prompt)
                group_data['images'].append(image)
                group_data['rewards'].append(reward)
                group_data['log_probs'].append(log_prob)
                group_data['ref_log_probs'].append(ref_log_prob)
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to process prompt '{prompt}': {e}")
                # 실패한 경우 기본값 사용
                group_data['enhanced_prompts'].append(prompt)
                group_data['images'].append(None)
                group_data['rewards'].append(0.0)
                group_data['log_probs'].append(torch.tensor(0.0))
                group_data['ref_log_probs'].append(torch.tensor(0.0))
        
        # 5. 어드밴티지 및 리턴 계산
        self._calculate_advantages_and_returns(group_data)
        
        logger.debug(f"✅ Group data collected: avg_reward={np.mean(group_data['rewards']):.4f}")
        return group_data
    
    def _enhance_prompt_with_logprob(self, prompt: str) -> Tuple[str, torch.Tensor]:
        """
        VLM으로 프롬프트 개선 및 로그 확률 계산
        
        Args:
            prompt (str): 원본 프롬프트
            
        Returns:
            Tuple[str, torch.Tensor]: (개선된 프롬프트, 로그 확률)
        """
        try:
            # VLM으로 프롬프트 개선 (실제 구현에서는 VLMWrapper 사용)
            enhanced_prompt = self.vlm.enhance_prompt(prompt)
            
            # 로그 확률 계산 (간소화된 버전)
            # 실제로는 생성된 토큰들의 로그 확률을 계산해야 함
            log_prob = torch.tensor(0.0, device=self.device)  # 플레이스홀더
            
            return enhanced_prompt, log_prob
            
        except Exception as e:
            logger.warning(f"⚠️ Prompt enhancement failed: {e}")
            return prompt, torch.tensor(0.0, device=self.device)
    
    def _calculate_reference_logprob(self, prompt: str, enhanced_prompt: str) -> torch.Tensor:
        """
        참조 모델로 로그 확률 계산
        
        Args:
            prompt (str): 원본 프롬프트
            enhanced_prompt (str): 개선된 프롬프트
            
        Returns:
            torch.Tensor: 참조 모델의 로그 확률
        """
        try:
            if self.vlm_ref is None:
                return torch.tensor(0.0, device=self.device)
            
            # 참조 모델로 로그 확률 계산 (간소화된 버전)
            ref_log_prob = torch.tensor(0.0, device=self.device)  # 플레이스홀더
            
            return ref_log_prob
            
        except Exception as e:
            logger.warning(f"⚠️ Reference log prob calculation failed: {e}")
            return torch.tensor(0.0, device=self.device)
    
    def _generate_image(self, prompt: str):
        """
        프롬프트로부터 이미지 생성 (플레이스홀더)
        
        실제 구현에서는 SD3Generator를 사용합니다.
        
        Args:
            prompt (str): 이미지 생성용 프롬프트
            
        Returns:
            생성된 이미지 (플레이스홀더)
        """
        # 실제 구현에서는 SD3Generator.generate_image() 호출
        return f"image_for_{prompt[:20]}"  # 플레이스홀더
    
    def _calculate_reward(self, image, enhanced_prompt: str, original_prompt: str) -> float:
        """
        이미지-텍스트 유사도 기반 보상 계산 (플레이스홀더)
        
        실제 구현에서는 CLIPRewardCalculator를 사용합니다.
        
        Args:
            image: 생성된 이미지
            enhanced_prompt (str): 개선된 프롬프트
            original_prompt (str): 원본 프롬프트
            
        Returns:
            float: 계산된 보상
        """
        # 실제 구현에서는 CLIPRewardCalculator.calculate_reward() 호출
        # 간단한 시뮬레이션: 프롬프트 길이에 따른 보상
        reward = min(len(enhanced_prompt) / 100.0, 1.0)
        return reward
    
    def _calculate_advantages_and_returns(self, group_data: Dict[str, Any]):
        """
        GRPO의 핵심: 그룹 기반 어드밴티지 계산
        
        GRPO는 그룹 내에서 상대적 성능을 비교하여 어드밴티지를 계산합니다:
        1. 그룹 평균 보상 계산
        2. 각 샘플의 상대적 성능 측정
        3. 할인된 리턴 계산
        4. 정규화
        
        Args:
            group_data (Dict[str, Any]): 그룹 데이터 (in-place 수정)
        """
        rewards = np.array(group_data['rewards'])
        
        # 1. 할인된 리턴 계산 (단순화: 단일 스텝)
        returns = rewards.copy()
        
        # 2. 그룹 기반 어드밴티지 계산
        # GRPO의 핵심: 그룹 평균 대비 상대적 성능
        group_mean_reward = np.mean(rewards)
        advantages = rewards - group_mean_reward
        
        # 3. 어드밴티지 정규화 (학습 안정성 향상)
        if len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # 4. 텐서로 변환
        group_data['returns'] = [torch.tensor(r, dtype=torch.float32, device=self.device) for r in returns]
        group_data['advantages'] = [torch.tensor(a, dtype=torch.float32, device=self.device) for a in advantages]
        
        logger.debug(f"📊 Advantages calculated: mean={np.mean(advantages):.4f}, std={np.std(advantages):.4f}")
    
    def _update_reference_model(self):
        """
        참조 모델 업데이트 (현재 정책의 복사본)
        
        GRPO에서 참조 모델은 KL 발산 제한을 위해 사용됩니다.
        매 iteration마다 현재 정책을 복사하여 참조 모델로 사용합니다.
        """
        try:
            # 현재 VLM의 깊은 복사본 생성
            self.vlm_ref = copy.deepcopy(self.vlm)
            self.vlm_ref.eval()  # 평가 모드로 설정
            
            # 참조 모델의 그래디언트 비활성화
            for param in self.vlm_ref.parameters():
                param.requires_grad = False
            
            logger.debug("🔄 Reference model updated")
            
        except Exception as e:
            logger.warning(f"⚠️ Reference model update failed: {e}")
    
    def grpo_update(self, group_data: Dict[str, Any]) -> Dict[str, float]:
        """
        GRPO 정책 업데이트 수행
        
        이 메서드는 GRPO 논문의 핵심 알고리즘을 구현합니다:
        1. 정책 비율 계산 (π_θ / π_ref)
        2. 클리핑된 서로게이트 손실
        3. KL 발산 페널티
        4. 엔트로피 보너스
        
        Args:
            group_data (Dict[str, Any]): 수집된 그룹 데이터
            
        Returns:
            Dict[str, float]: 학습 메트릭
        """
        metrics = {
            'policy_loss': 0.0,
            'kl_div': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0
        }
        
        # GRPO 에포크만큼 반복 학습
        for epoch in range(self.config.grpo_epochs):
            epoch_metrics = self._grpo_epoch_update(group_data)
            
            # 메트릭 누적
            for key in metrics:
                metrics[key] += epoch_metrics[key]
        
        # 평균 계산
        for key in metrics:
            metrics[key] /= self.config.grpo_epochs
        
        # 학습 통계 업데이트
        self.training_stats.update(metrics)
        self.training_stats['iteration'] += 1
        self.training_stats['total_samples'] += len(group_data['prompts'])
        self.training_stats['avg_reward'] = np.mean(group_data['rewards'])
        
        logger.info(f"🔄 GRPO update completed: loss={metrics['total_loss']:.4f}, "
                   f"reward={self.training_stats['avg_reward']:.4f}")
        
        return metrics
    
    def _grpo_epoch_update(self, group_data: Dict[str, Any]) -> Dict[str, float]:
        """
        단일 GRPO 에포크 업데이트
        
        Args:
            group_data (Dict[str, Any]): 그룹 데이터
            
        Returns:
            Dict[str, float]: 에포크 메트릭
        """
        self.optimizer.zero_grad()
        
        # 현재 정책으로 로그 확률 재계산 (정책이 업데이트되었으므로)
        current_log_probs = []
        for i, prompt in enumerate(group_data['prompts']):
            enhanced_prompt = group_data['enhanced_prompts'][i]
            _, log_prob = self._enhance_prompt_with_logprob(prompt)
            current_log_probs.append(log_prob)
        
        # 손실 계산
        policy_loss = 0.0
        kl_div = 0.0
        entropy = 0.0
        
        for i in range(len(group_data['prompts'])):
            # 정책 비율 계산: π_θ(a|s) / π_ref(a|s)
            log_ratio = current_log_probs[i] - group_data['ref_log_probs'][i]
            ratio = torch.exp(log_ratio)
            
            # 어드밴티지
            advantage = group_data['advantages'][i]
            
            # 클리핑된 서로게이트 손실 (PPO 스타일)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantage
            policy_loss_i = -torch.min(surr1, surr2)
            
            # KL 발산 (근사치)
            kl_div_i = log_ratio  # 근사: log(π/π_ref) ≈ KL(π_ref, π)
            
            # 엔트로피 (간소화)
            entropy_i = -current_log_probs[i]  # 간소화된 엔트로피
            
            policy_loss += policy_loss_i
            kl_div += kl_div_i
            entropy += entropy_i
        
        # 배치 크기로 정규화
        batch_size = len(group_data['prompts'])
        policy_loss /= batch_size
        kl_div /= batch_size
        entropy /= batch_size
        
        # 총 손실: 정책 손실 + KL 페널티 - 엔트로피 보너스
        total_loss = policy_loss + self.config.kl_beta * kl_div - self.config.entropy_coeff * entropy
        
        # 역전파
        total_loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.vlm.parameters(), self.config.max_grad_norm)
        
        # 옵티마이저 스텝
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'kl_div': kl_div.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """현재 학습 통계 반환"""
        return self.training_stats.copy()
    
    def save_checkpoint(self, checkpoint_path: str):
        """학습 체크포인트 저장"""
        try:
            checkpoint = {
                'model_state_dict': self.vlm.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'training_stats': self.training_stats
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """학습 체크포인트 로드"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.vlm.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_stats = checkpoint['training_stats']
            logger.info(f"📥 Checkpoint loaded: {checkpoint_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")


if __name__ == "__main__":
    # GRPO Trainer 테스트 코드
    print("🧪 GRPO Trainer Test")
    print("=" * 30)
    
    try:
        # Mock VLM 모델
        class MockVLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def enhance_prompt(self, prompt):
                return f"enhanced: {prompt}"
        
        # 설정 및 트레이너 초기화
        config = GRPOConfig(
            learning_rate=1e-4,
            group_size=3,
            num_iterations=2
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
    
    print("\nUsage:")
    print("from training.grpo_trainer import GRPOTrainer, GRPOConfig")
    print("config = GRPOConfig()")
    print("trainer = GRPOTrainer(vlm_model, config)")
    print("group_data = trainer.collect_group_data(prompts)")
    print("metrics = trainer.grpo_update(group_data)") 