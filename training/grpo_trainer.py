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
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import logging
from transformers import AutoTokenizer

# 로깅 설정
logging.basicConfig(level=logging.INFO)
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
    
    # 토큰 생성 파라미터
    max_new_tokens: int = 20      # 최대 생성 토큰 수
    vocab_size: int = 50000       # 어휘 크기
    max_sequence_length: int = 100 # 최대 시퀀스 길이
    temperature: float = 0.8
    
    # 디바이스 설정
    device: str = "auto"

class TokenPolicyNetwork(nn.Module):
    """토큰별 정책 네트워크 - 다음 토큰을 선택하는 모델"""
    
    def __init__(self, vocab_size: int = 50000, embed_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # 토큰 임베딩
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(100, embed_dim)  # 최대 100 토큰
        
        # 트랜스포머 레이어
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )
        
        # 출력 헤드
        self.output_head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Args:
            input_ids: [batch_size, seq_len] - 토큰 ID 시퀀스
            attention_mask: [batch_size, seq_len] - 어텐션 마스크
            
        Returns:
            Categorical: 다음 토큰에 대한 확률 분포
        """
        batch_size, seq_len = input_ids.shape
        
        # 토큰 + 위치 임베딩
        token_embeds = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)
        
        # 임베딩 합성
        embeddings = token_embeds + pos_embeds
        
        # 트랜스포머 처리
        if attention_mask is not None:
            # 패딩 마스크 생성 (True = 마스킹)
            padding_mask = (attention_mask == 0)
        else:
            padding_mask = None
        
        transformer_output = self.transformer(embeddings, src_key_padding_mask=padding_mask)
        
        # 마지막 토큰의 출력을 사용하여 다음 토큰 예측
        last_token_output = transformer_output[:, -1, :]  # [batch_size, embed_dim]
        logits = self.output_head(last_token_output)      # [batch_size, vocab_size]
        
        # 확률 분포 반환
        return Categorical(logits=logits)

class GRPOTrainer:
    """토큰별 순차 생성 기반 GRPO 트레이너"""
    
    def __init__(self, vlm_model, sd_generator, clip_reward, config: GRPOConfig):
        self.vlm = vlm_model
        self.sd_generator = sd_generator  # 동결된 SD3 파이프라인
        self.clip_reward = clip_reward    # 동결된 CLIP 보상 모델
        self.config = config
        
        # 디바이스 설정
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        # 토크나이저 초기화
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 정책 네트워크 초기화
        self.policy_network = TokenPolicyNetwork(
            vocab_size=len(self.tokenizer),
            embed_dim=256,
            hidden_dim=512
        ).to(self.device)
        
        # 옵티마이저
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
        
        logger.info(f"🚀 GRPO Trainer initialized with device: {self.device}")
        logger.info(f"📊 Policy network parameters: {sum(p.numel() for p in self.policy_network.parameters())}")
        logger.info(f"📝 Tokenizer vocab size: {len(self.tokenizer)}")
    
    def collect_group_data(self, prompts: List[str]) -> Dict[str, Any]:
        """
        그룹 데이터 수집 - 토큰별 순차 생성 방식
        
        각 프롬프트에 대해:
        1. 토큰별로 순차 생성
        2. 각 스텝에서 state = user_prompt + 지금까지_생성된_토큰들
        3. action = 다음_토큰_선택
        4. 완성된 프롬프트로 이미지 생성 및 보상 계산
        """
        logger.info(f"🔄 Collecting group data for {len(prompts)} prompts...")
        
        group_data = {
            'original_prompts': [],
            'generated_sequences': [],  # 생성된 전체 시퀀스
            'states': [],              # 각 스텝의 상태
            'actions': [],             # 각 스텝의 액션 (토큰)
            'log_probs_old': [],       # 각 스텝의 로그 확률
            'rewards': [],             # 각 시퀀스의 최종 보상
            'episode_lengths': []      # 각 에피소드 길이
        }
        
        self.policy_network.eval()
        
        for prompt in prompts:
            logger.debug(f"📝 Processing prompt: '{prompt[:50]}...'")
            
            # 1. 프롬프트 토크나이징
            initial_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # 2. 토큰별 순차 생성
            episode_states = []
            episode_actions = []
            episode_log_probs = []
            
            current_sequence = initial_tokens.clone()
            
            with torch.no_grad():
                for step in range(self.config.max_new_tokens):
                    # 현재 상태: user_prompt + 지금까지_생성된_토큰들
                    current_state = current_sequence.clone()
                    
                    # 정책 네트워크로 다음 토큰 분포 계산
                    policy_dist = self.policy_network(current_sequence)
                    
                    # 다음 토큰 샘플링
                    next_token = policy_dist.sample()
                    log_prob = policy_dist.log_prob(next_token)
                    
                    # 데이터 저장
                    episode_states.append(current_state.squeeze())
                    episode_actions.append(next_token)
                    episode_log_probs.append(log_prob)
                    
                    # 시퀀스 업데이트
                    current_sequence = torch.cat([current_sequence, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                    
                    # EOS 토큰이면 중단
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
            
            # 3. 생성된 프롬프트 디코딩
            generated_sequence = current_sequence.squeeze()
            generated_text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
            
            # 4. 환경 실행: 동결된 텍스트→이미지 파이프라인
            try:
                # SD3로 이미지 생성
                generated_image = self.sd_generator.generate_image(generated_text)
                
                # CLIP으로 보상 계산
                reward = self.clip_reward.calculate_reward(
                    image=generated_image,
                    text=generated_text,
                    original_prompt=prompt
                )
            except Exception as e:
                logger.warning(f"⚠️ Image generation/reward failed: {e}")
                reward = 0.0
            
            # 5. 그룹 데이터에 추가
            group_data['original_prompts'].append(prompt)
            group_data['generated_sequences'].append(generated_sequence)
            group_data['states'].extend(episode_states)
            group_data['actions'].extend(episode_actions)
            group_data['log_probs_old'].extend(episode_log_probs)
            
            # 각 스텝에 동일한 최종 보상 할당 (할인 적용 예정)
            episode_length = len(episode_states)
            group_data['rewards'].extend([reward] * episode_length)
            group_data['episode_lengths'].append(episode_length)
            
            logger.debug(f"✅ Generated: '{generated_text[:50]}...', Reward: {reward:.4f}, Length: {episode_length}")
        
        logger.info(f"📊 Collected {len(group_data['states'])} steps from {len(prompts)} episodes")
        logger.info(f"📊 Average reward: {np.mean(group_data['rewards']):.4f}")
        
        return group_data
    
    def _calculate_advantages_and_returns(self, group_data: Dict[str, Any]):
        """
        할인된 리턴과 어드밴티지 계산 (에피소드별)
        """
        logger.debug("🔄 Calculating discounted returns and advantages...")
        
        returns = []
        advantages = []
        
        start_idx = 0
        for episode_length in group_data['episode_lengths']:
            # 에피소드 보상 추출
            episode_rewards = group_data['rewards'][start_idx:start_idx + episode_length]
            
            # 할인된 리턴 계산 (역순)
            episode_returns = []
            discounted_return = 0.0
            
            for reward in reversed(episode_rewards):
                discounted_return = reward + self.config.gamma * discounted_return
                episode_returns.insert(0, discounted_return)
            
            returns.extend(episode_returns)
            start_idx += episode_length
        
        # 그룹 정규화를 위한 어드밴티지 계산
        returns_array = np.array(returns, dtype=np.float32)
        
        if len(returns_array) > 1:
            group_mean = np.mean(returns_array)
            group_std = np.std(returns_array)
            advantages_array = (returns_array - group_mean) / (group_std + self.config.epsilon_std)
        else:
            advantages_array = np.array([0.0])
        
        # 텐서로 변환
        group_data['returns'] = [torch.tensor(ret, dtype=torch.float32, device=self.device) for ret in returns]
        group_data['advantages'] = [torch.tensor(adv, dtype=torch.float32, device=self.device) for adv in advantages_array]
        
        logger.debug(f"📊 Returns: mean={np.mean(returns_array):.4f}, std={np.std(returns_array):.4f}")
        logger.debug(f"📊 Advantages: mean={np.mean(advantages_array):.4f}, std={np.std(advantages_array):.4f}")
    
    def grpo_update(self, group_data: Dict[str, Any]) -> Dict[str, float]:
        """
        GRPO 업데이트 - 토큰별 정책 개선
        """
        logger.info("🔄 Starting GRPO update...")
        
        # 1. 어드밴티지 계산
        self._calculate_advantages_and_returns(group_data)
        
        # 2. 참조 모델 생성
        policy_ref = TokenPolicyNetwork(
            vocab_size=len(self.tokenizer),
            embed_dim=256,
            hidden_dim=512
        ).to(self.device)
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
        
        # 4. 평균 메트릭
        avg_metrics = {
            'policy_loss': total_policy_loss / self.config.grpo_epochs,
            'entropy': total_entropy / self.config.grpo_epochs,
            'kl_div': total_kl_div / self.config.grpo_epochs,
            'avg_reward': np.mean([r for episode_rewards in group_data['rewards'] for r in episode_rewards] if isinstance(group_data['rewards'][0], list) else group_data['rewards'])
        }
        
        # 5. 통계 업데이트
        self.training_stats.update(avg_metrics)
        self.training_stats['iteration'] += 1
        self.training_stats['total_samples'] += len(group_data['states'])
        
        logger.info(f"✅ GRPO update complete:")
        logger.info(f"  Policy Loss: {avg_metrics['policy_loss']:.4f}")
        logger.info(f"  Entropy: {avg_metrics['entropy']:.4f}")
        logger.info(f"  KL Div: {avg_metrics['kl_div']:.4f}")
        logger.info(f"  Avg Reward: {avg_metrics['avg_reward']:.4f}")
        
        return avg_metrics
    
    def _grpo_epoch_update(self, group_data: Dict[str, Any], policy_ref: TokenPolicyNetwork) -> Dict[str, float]:
        """
        단일 에포크 GRPO 업데이트
        """
        self.optimizer.zero_grad()
        
        total_policy_loss = 0.0
        total_entropy = 0.0
        total_kl_div = 0.0
        batch_size = len(group_data['states'])
        
        for i in range(batch_size):
            # 현재 스텝 데이터
            state = group_data['states'][i].unsqueeze(0)  # [1, seq_len]
            action = group_data['actions'][i]
            old_log_prob = group_data['log_probs_old'][i]
            advantage = group_data['advantages'][i]
            
            # 현재 정책 분포
            current_policy_dist = self.policy_network(state)
            current_log_prob = current_policy_dist.log_prob(action)
            
            # 참조 정책 분포
            with torch.no_grad():
                ref_policy_dist = policy_ref(state)
            
            # PPO/GRPO 손실 계산
            log_ratio = current_log_prob - old_log_prob.detach()
            ratio = torch.exp(log_ratio)
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantage
            policy_loss_i = -torch.min(surr1, surr2)
            
            # 엔트로피 및 KL 발산
            entropy_i = current_policy_dist.entropy()
            kl_div_i = torch.distributions.kl_divergence(ref_policy_dist, current_policy_dist)
            
            total_policy_loss += policy_loss_i
            total_entropy += entropy_i
            total_kl_div += kl_div_i
        
        # 평균 및 총 손실
        avg_policy_loss = total_policy_loss / batch_size
        avg_entropy = total_entropy / batch_size
        avg_kl_div = total_kl_div / batch_size
        
        total_loss = avg_policy_loss + self.config.kl_beta * avg_kl_div - self.config.entropy_coeff * avg_entropy
        
        # 역전파
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.config.max_grad_norm)
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
                'tokenizer': self.tokenizer,
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
            logger.info(f"📂 Checkpoint loaded: {checkpoint_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")

# 테스트용 Mock 클래스들
if __name__ == "__main__":
    class MockVLM:
        def enhance_prompt(self, prompt):
            return f"enhanced {prompt}"
    
    class MockSDGenerator:
        def generate_image(self, prompt):
            return f"image_for_{prompt[:20]}"
    
    class MockCLIPReward:
        def calculate_reward(self, image, text, original_prompt):
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
    group_data = trainer.collect_group_data(test_prompts)
    metrics = trainer.grpo_update(group_data)
    print(f"Test metrics: {metrics}") 