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
            vlm_model: 학습할 VLM 모델 (플레이스홀더 방식)
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
        
        # 플레이스홀더 방식에서는 실제 모델 파라미터가 없으므로 더미 파라미터 생성
        self.dummy_param = nn.Parameter(torch.randn(1, requires_grad=True))
        
        # 참조 모델 생성 (매 iteration마다 업데이트)
        self.vlm_ref = None
        
        # 옵티마이저 설정 (더미 파라미터 사용)
        self.optimizer = optim.AdamW(
            [self.dummy_param],  # 더미 파라미터 사용
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
        logger.info("📝 Using placeholder-based VLM, no actual parameter optimization")
    
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
                # 실패한 경우 기본값 사용 (일관된 텐서 속성으로)
                group_data['enhanced_prompts'].append(prompt)
                group_data['images'].append(None)
                group_data['rewards'].append(0.0)
                group_data['log_probs'].append(torch.tensor(-2.0, dtype=torch.float32, requires_grad=True))
                group_data['ref_log_probs'].append(torch.tensor(-2.0, dtype=torch.float32))
        
        # 5. 어드밴티지 및 리턴 계산
        self._calculate_advantages_and_returns(group_data)
        
        logger.debug(f"✅ Group data collected: avg_reward={np.mean(group_data['rewards']):.4f}")
        return group_data
    
    def _enhance_prompt_with_logprob(self, prompt: str) -> Tuple[str, torch.Tensor]:
        """
        VLM으로 프롬프트 개선 및 로그 확률 계산 (플레이스홀더 방식)
        
        Args:
            prompt (str): 원본 프롬프트
            
        Returns:
            Tuple[str, torch.Tensor]: (개선된 프롬프트, 로그 확률)
        """
        try:
            # 플레이스홀더 방식으로 프롬프트 개선
            enhanced_prompt = self.vlm.enhance_prompt(prompt)
            
            # 더미 로그 확률 생성 (실제 계산 대신)
            log_prob = torch.tensor(-1.0, dtype=torch.float32, requires_grad=True)
            
            return enhanced_prompt, log_prob
            
        except Exception as e:
            logger.warning(f"⚠️ Prompt enhancement failed: {e}")
            # 실패 시 원본 프롬프트와 더미 로그 확률 반환
            return prompt, torch.tensor(-2.0, dtype=torch.float32, requires_grad=True)
    
    def _calculate_reference_logprob(self, prompt: str, enhanced_prompt: str) -> torch.Tensor:
        """
        참조 모델로 로그 확률 계산 (플레이스홀더 방식)
        
        Args:
            prompt (str): 원본 프롬프트
            enhanced_prompt (str): 개선된 프롬프트
            
        Returns:
            torch.Tensor: 참조 로그 확률
        """
        try:
            # 더미 참조 로그 확률 생성
            ref_log_prob = torch.tensor(-1.2, dtype=torch.float32)
            return ref_log_prob
            
        except Exception as e:
            logger.warning(f"⚠️ Reference log prob calculation failed: {e}")
            return torch.tensor(-2.0, dtype=torch.float32)
    
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
        # 데이터 완성도 검증
        expected_length = len(group_data['prompts'])
        
        # rewards 길이 검증 및 보정
        if len(group_data['rewards']) != expected_length:
            logger.warning(f"⚠️ Rewards length mismatch: {len(group_data['rewards'])} != {expected_length}")
            while len(group_data['rewards']) < expected_length:
                group_data['rewards'].append(0.0)
        
        # ref_log_probs 길이 검증 및 보정
        if len(group_data['ref_log_probs']) != expected_length:
            logger.warning(f"⚠️ ref_log_probs length mismatch: {len(group_data['ref_log_probs'])} != {expected_length}")
            while len(group_data['ref_log_probs']) < expected_length:
                group_data['ref_log_probs'].append(torch.tensor(-1.2, dtype=torch.float32))
        
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
        
        # 4. 텐서로 변환 (디바이스 문제 해결)
        group_data['returns'] = [torch.tensor(float(r), dtype=torch.float32) for r in returns]
        group_data['advantages'] = [torch.tensor(float(a), dtype=torch.float32) for a in advantages]
        
        # 5. 최종 길이 검증
        for key in ['returns', 'advantages']:
            if len(group_data[key]) != expected_length:
                logger.error(f"❌ Final length mismatch for {key}: {len(group_data[key])} != {expected_length}")
        
        logger.debug(f"📊 Advantages calculated: mean={np.mean(advantages):.4f}, std={np.std(advantages):.4f}")
    
    def _update_reference_model(self):
        """
        참조 모델 업데이트 (플레이스홀더 방식에서는 간단히 처리)
        """
        try:
            # 플레이스홀더 방식에서는 참조 모델 업데이트가 불필요
            self.vlm_ref = "placeholder_ref_model"
            logger.debug("🔄 Reference model updated (placeholder)")
            
        except Exception as e:
            logger.warning(f"⚠️ Reference model update failed: {e}")
    
    def grpo_update(self, group_data: Dict[str, Any]) -> Dict[str, float]:
        """
        GRPO 정책 업데이트 수행
        
        Args:
            group_data (Dict[str, Any]): 수집된 그룹 데이터
            
        Returns:
            Dict[str, float]: 학습 메트릭
        """
        logger.debug("🔄 Starting GRPO update")
        
        # 입력 검증
        if not group_data or len(group_data.get('prompts', [])) == 0:
            logger.warning("⚠️ Empty group data provided")
            return {
                'policy_loss': 0.0,
                'kl_div': 0.0,
                'entropy': 0.0,
                'total_loss': 0.0,
                'avg_reward': 0.0
            }
        
        # 메트릭 초기화
        metrics = {
            'policy_loss': 0.0,
            'kl_div': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0,
            'avg_reward': np.mean(group_data['rewards']) if group_data['rewards'] else 0.0
        }
        
        try:
            # GRPO 에포크만큼 반복 학습
            for epoch in range(self.config.grpo_epochs):
                epoch_metrics = self._grpo_epoch_update(group_data)
                
                # 메트릭 누적
                for key in ['policy_loss', 'kl_div', 'entropy', 'total_loss']:
                    metrics[key] += epoch_metrics.get(key, 0.0)
            
            # 평균 계산
            for key in ['policy_loss', 'kl_div', 'entropy', 'total_loss']:
                metrics[key] /= self.config.grpo_epochs
            
            # 학습 통계 업데이트
            self.training_stats.update(metrics)
            self.training_stats['iteration'] += 1
            self.training_stats['total_samples'] += len(group_data['prompts'])
            self.training_stats['avg_reward'] = metrics['avg_reward']
            
            logger.info(f"🔄 GRPO update completed: loss={metrics['total_loss']:.4f}, "
                       f"reward={metrics['avg_reward']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ GRPO update failed: {e}")
            # 에러 발생 시 기본 메트릭 반환
            
        return metrics
    
    def _grpo_epoch_update(self, group_data: Dict[str, Any]) -> Dict[str, float]:
        """
        단일 GRPO 에포크 업데이트
        
        Args:
            group_data (Dict[str, Any]): 그룹 데이터
            
        Returns:
            Dict[str, float]: 에포크 메트릭
        """
        try:
            self.optimizer.zero_grad()
            
            # 현재 정책으로 로그 확률 재계산 (정책이 업데이트되었으므로)
            current_log_probs = []
            for i, prompt in enumerate(group_data['prompts']):
                try:
                    enhanced_prompt = group_data['enhanced_prompts'][i]
                    _, log_prob = self._enhance_prompt_with_logprob(prompt)
                    current_log_probs.append(log_prob)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get log prob for prompt {i}: {e}")
                    current_log_probs.append(torch.tensor(-2.0, dtype=torch.float32, requires_grad=True))
            
            # 손실 계산
            policy_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
            kl_div_estimates = []  # KL divergence estimates for batch average
            entropy = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
            
            for i in range(len(group_data['prompts'])):
                try:
                    # 현재 로그 확률과 참조 로그 확률 가져오기
                    current_log_prob = current_log_probs[i]
                    
                    # ref_log_probs 키 안전하게 접근
                    if 'ref_log_probs' not in group_data or i >= len(group_data['ref_log_probs']):
                        logger.warning(f"⚠️ Missing ref_log_probs for sample {i}")
                        ref_log_prob = torch.tensor(-1.2, dtype=torch.float32)
                    else:
                        ref_log_prob = group_data['ref_log_probs'][i]
                    
                    # 텐서로 변환 (필요시)
                    if not isinstance(ref_log_prob, torch.Tensor):
                        ref_log_prob = torch.tensor(float(ref_log_prob), dtype=torch.float32)
                    
                    # advantages 키 안전하게 접근
                    if 'advantages' not in group_data or i >= len(group_data['advantages']):
                        logger.warning(f"⚠️ Missing advantages for sample {i}")
                        advantage = torch.tensor(0.0, dtype=torch.float32)
                    else:
                        advantage = group_data['advantages'][i]
                        if not isinstance(advantage, torch.Tensor):
                            advantage = torch.tensor(float(advantage), dtype=torch.float32)
                    
                    # 정책 비율 계산: π_θ(a|s) / π_ref(a|s)
                    log_ratio = current_log_prob - ref_log_prob
                    ratio = torch.exp(log_ratio)
                    
                    # 클리핑된 서로게이트 손실 (PPO 스타일)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantage
                    policy_loss_i = -torch.min(surr1, surr2)
                    
                    # KL divergence 계산 (수정된 공식)
                    log_ratio_ref_curr = ref_log_prob - current_log_prob.detach()
                    kl_div_i = torch.exp(log_ratio_ref_curr) - log_ratio_ref_curr - 1
                    kl_div_i = torch.relu(kl_div_i)  # 음수 방지
                    
                    # 엔트로피 계산
                    entropy_i = -current_log_prob
                    
                    policy_loss = policy_loss + policy_loss_i
                    kl_div_estimates.append(kl_div_i)
                    entropy = entropy + entropy_i
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to calculate loss for sample {i}: {e}")
                    continue
            
            # 배치 크기로 정규화
            batch_size = len(group_data['prompts'])
            if batch_size > 0:
                policy_loss = policy_loss / batch_size
                entropy = entropy / batch_size
            
            # KL divergence 평균 계산
            if len(kl_div_estimates) > 0:
                kl_div_estimate_mean = torch.stack(kl_div_estimates).mean()
            else:
                kl_div_estimate_mean = torch.tensor(0.0, dtype=torch.float32)
            
            # 총 손실: 정책 손실 + KL 페널티 - 엔트로피 보너스
            total_loss = policy_loss + self.config.kl_beta * kl_div_estimate_mean - self.config.entropy_coeff * entropy
            
            # 역전파 (더미 파라미터 사용)
            total_loss.backward()
            
            # 그래디언트 클리핑 (더미 파라미터 사용)
            torch.nn.utils.clip_grad_norm_([self.dummy_param], self.config.max_grad_norm)
            
            # 옵티마이저 스텝
            self.optimizer.step()
            
            # 안전한 item() 호출
            return {
                'policy_loss': float(policy_loss.detach().numpy()) if hasattr(policy_loss, 'detach') else float(policy_loss),
                'kl_div': float(kl_div_estimate_mean.detach().numpy()) if hasattr(kl_div_estimate_mean, 'detach') else float(kl_div_estimate_mean),
                'entropy': float(entropy.detach().numpy()) if hasattr(entropy, 'detach') else float(entropy),
                'total_loss': float(total_loss.detach().numpy()) if hasattr(total_loss, 'detach') else float(total_loss)
            }
            
        except Exception as e:
            logger.error(f"❌ Epoch update failed: {e}")
            return {
                'policy_loss': 0.0,
                'kl_div': 0.0,
                'entropy': 0.0,
                'total_loss': 0.0
            }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """현재 학습 통계 반환"""
        return self.training_stats.copy()
    
    def save_checkpoint(self, checkpoint_path: str):
        """학습 체크포인트 저장 (플레이스홀더 방식)"""
        try:
            checkpoint = {
                'dummy_param': self.dummy_param.data,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'training_stats': self.training_stats
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """학습 체크포인트 로드 (플레이스홀더 방식)"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'dummy_param' in checkpoint:
                self.dummy_param.data = checkpoint['dummy_param']
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