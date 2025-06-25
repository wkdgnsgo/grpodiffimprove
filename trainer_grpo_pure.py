#!/usr/bin/env python3
"""
순수 GRPO 트레이너 (easyr1 스타일)
Value Network 없이 오직 Policy Network만 사용
그룹 평균을 implicit baseline으로 사용하는 올바른 GRPO 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class PureGRPOConfig:
    """순수 GRPO 설정 (Value Network 없음)"""
    learning_rate: float = 1e-6
    batch_size: int = 4
    num_rollouts: int = 5  # 그룹별 롤아웃 수
    max_prompt_length: int = 77
    max_new_tokens: int = 30
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 100
    kl_coef: float = 0.02
    clip_ratio: float = 0.1
    entropy_coef: float = 0.02
    vocab_size: int = 32000

class PureGRPOPolicy(nn.Module):
    """순수 GRPO 정책 네트워크 (Value Head 없음)"""
    
    def __init__(self, qwen_model, config: PureGRPOConfig):
        super().__init__()
        self.qwen_model = qwen_model
        self.config = config
        
        # GPU 디바이스 설정
        self.qwen_device = "cuda:0"  # QWEN은 GPU 0
        self.policy_device = "cuda:0"  # Policy head도 GPU 0에서 학습
        
        self.hidden_size = qwen_model.model.config.hidden_size
        self.vocab_size = len(qwen_model.tokenizer.get_vocab())
        
        logger.info(f"순수 GRPO 정책 - Hidden: {self.hidden_size}, Vocab: {self.vocab_size}")
        logger.info(f"GPU 배치: QWEN={self.qwen_device}, Policy={self.policy_device}")
        
        # 오직 정책 헤드만! (Value Head 없음) - GPU 0에 배치
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_size, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, self.vocab_size)
        ).to(self.policy_device)
        
        self._init_weights()
        
        logger.info(f"순수 GRPO 정책 네트워크 초기화 완료 - Action Space: {self.vocab_size}")
        logger.info("✅ Value Network 없음 - 그룹 평균을 implicit baseline으로 사용")
    
    def _init_weights(self):
        """가중치 초기화"""
        for layer in self.policy_head:
            if isinstance(layer, nn.Linear):
                gain = 0.02 if layer.out_features == self.vocab_size else 0.1
                nn.init.xavier_normal_(layer.weight, gain=gain)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """오직 정책 로짓만 반환 (Values 없음)"""
        batch_size = input_ids.size(0)
        
        # 입력 텐서를 QWEN GPU(0번)로 이동
        input_ids = input_ids.to(self.qwen_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.qwen_device)
        
        with torch.no_grad():
            outputs = self.qwen_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # QWEN2VL 모델은 last_hidden_state 대신 hidden_states를 사용
            if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                hidden_states = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                # hidden_states는 튜플이므로 마지막 레이어 선택
                hidden_states = outputs.hidden_states[-1]
            else:
                # 대안: logits에서 히든 스테이트 추출 시도
                raise AttributeError("Cannot find hidden states in model output")
        
        if attention_mask is not None:
            last_valid_indices = attention_mask.sum(dim=1) - 1
            last_valid_indices = torch.clamp(last_valid_indices, min=0)
            last_hidden = hidden_states[torch.arange(batch_size, device=self.qwen_device), last_valid_indices]
        else:
            last_hidden = hidden_states[:, -1, :]
        
        # Hidden states를 Policy GPU로 이동 (GPU 0이므로 동일하지만 명시적으로)
        last_hidden = last_hidden.to(self.policy_device)
        
        # 오직 정책 로짓만 반환!
        policy_logits = self.policy_head(last_hidden)
        
        return policy_logits  # Values 없음!
    
    def get_action_and_log_prob(self, state: Dict):
        """액션 선택과 로그 확률 (Value 없음)"""
        input_ids = state['input_ids'].unsqueeze(0)
        attention_mask = state['attention_mask'].unsqueeze(0)
        
        policy_logits = self(input_ids, attention_mask)
        
        scaled_logits = policy_logits / self.config.temperature
        scaled_logits = torch.clamp(scaled_logits, min=-10, max=10)
        
        # Top-k 필터링
        if self.config.top_k > 0:
            top_k_logits, top_k_indices = torch.topk(scaled_logits, self.config.top_k, dim=-1)
            scaled_logits = torch.full_like(scaled_logits, float('-inf'))
            scaled_logits.scatter_(-1, top_k_indices, top_k_logits)
        
        token_probs = F.softmax(scaled_logits, dim=-1)
        
        # Top-p 필터링
        if self.config.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(token_probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            token_probs = token_probs.masked_fill(indices_to_remove, 0.0)
            
            prob_sum = token_probs.sum(dim=-1, keepdim=True)
            token_probs = token_probs / (prob_sum + 1e-8)
        
        try:
            token_dist = torch.distributions.Categorical(token_probs)
            action = token_dist.sample()
            action_log_prob = token_dist.log_prob(action)
        except ValueError:
            logger.warning("Invalid probability distribution, using uniform sampling")
            action = torch.randint(0, self.vocab_size, (1,))
            action_log_prob = torch.log(torch.tensor(1.0 / self.vocab_size))
        
        # Value 없음! 오직 action, log_prob, logits만 반환
        return action.item(), action_log_prob, scaled_logits.squeeze(0)

class PureGRPOPromptEnvironment:
    """순수 GRPO용 프롬프트 환경"""
    
    def __init__(self, qwen_model, reward_model, sd_pipeline, config: PureGRPOConfig):
        self.qwen_model = qwen_model
        self.reward_model = reward_model
        self.sd_pipeline = sd_pipeline
        self.config = config
        self.tokenizer = qwen_model.tokenizer
        self.vocab_size = len(self.tokenizer.get_vocab())
        
        # GPU 디바이스 설정
        self.qwen_device = "cuda:0"  # QWEN (토큰화)
        self.sd_device = "cuda:1"    # Stable Diffusion (이미지 생성)
        self.reward_device = "cuda:2"  # CLIP Reward (리워드 계산)
        
        self.current_prompt = ""
        self.original_prompt = ""
        self.step_count = 0
        
        logger.info(f"순수 GRPO 환경 초기화 - Vocab: {self.vocab_size}")
        logger.info(f"GPU 배치: QWEN={self.qwen_device}, SD={self.sd_device}, Reward={self.reward_device}")
    
    def reset(self, user_prompt: str):
        """환경 리셋 - GPU 0으로 토큰 이동"""
        self.original_prompt = user_prompt
        self.current_prompt = user_prompt
        self.step_count = 0
        
        # 현재 프롬프트를 토큰화하고 QWEN GPU(0번)로 이동
        tokens = self.tokenizer.encode(
            self.current_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
            padding=False
        )
        
        attention_mask = torch.ones_like(tokens)
        
        return {
            'input_ids': tokens.squeeze(0).to(self.qwen_device),
            'attention_mask': attention_mask.squeeze(0).to(self.qwen_device)
        }
    
    def step(self, action: int):
        """환경 스텝 - GPU 간 데이터 이동 처리"""
        # 액션(토큰)을 텍스트로 변환
        try:
            token_text = self.tokenizer.decode([action], skip_special_tokens=True)
            
            # 프롬프트에 토큰 추가
            if token_text.strip():
                if self.current_prompt.endswith(' ') or token_text.startswith(' '):
                    self.current_prompt += token_text
                else:
                    self.current_prompt += ' ' + token_text
            
            self.step_count += 1
            
            # 종료 조건
            done = (self.step_count >= self.config.max_new_tokens or 
                   action == self.tokenizer.eos_token_id or
                   len(self.current_prompt) >= self.config.max_prompt_length * 4)
            
            # 리워드 계산 (에피소드 끝에만) - GPU 간 이동 처리
            if done:
                try:
                    logger.info(f"🖼️  이미지 생성 시작 (GPU {self.sd_device})")
                    
                    # SD3 파이프라인을 GPU 1로 이동하여 이미지 생성
                    with torch.cuda.device(1):
                        result = self.sd_pipeline(
                            prompt=self.current_prompt,
                            num_inference_steps=20,
                            guidance_scale=7.0,
                            height=512,
                            width=512
                        )
                        image = result.images[0]
                    
                    logger.info(f"🎯 리워드 계산 시작 (GPU {self.reward_device})")
                    
                    # CLIP 리워드를 GPU 2에서 계산
                    with torch.cuda.device(2):
                        reward = self.reward_model.calculate_reward(
                            self.original_prompt,
                            self.current_prompt,
                            image
                        )
                    
                    # 길이 보너스
                    length_bonus = min(self.step_count / self.config.max_new_tokens, 1.0) * 0.5
                    total_reward = reward + length_bonus
                    
                    logger.info(f"✅ 리워드 계산 완료: {total_reward:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Reward calculation failed: {e}")
                    total_reward = 0.0
            else:
                total_reward = 0.0
            
            # 다음 상태 (GPU 0으로 이동)
            if not done:
                next_tokens = self.tokenizer.encode(
                    self.current_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_prompt_length,
                    padding=False
                )
                next_attention_mask = torch.ones_like(next_tokens)
                
                # 다음 상태를 QWEN GPU(0번)로 이동
                next_state = {
                    'input_ids': next_tokens.squeeze(0).to(self.qwen_device),
                    'attention_mask': next_attention_mask.squeeze(0).to(self.qwen_device)
                }
            else:
                next_state = None
            
            info = {
                'current_prompt': self.current_prompt,
                'step_count': self.step_count,
                'token_added': token_text
            }
            
            return next_state, total_reward, done, info
            
        except Exception as e:
            logger.warning(f"Step failed: {e}")
            return None, 0.0, True, {'error': str(e)}

class PureGRPOTrainer:
    """순수 GRPO 트레이너 (Value Network 없음)"""
    
    def __init__(self, qwen_model, reward_model, sd_pipeline, config: PureGRPOConfig):
        self.config = config
        self.qwen_model = qwen_model
        self.reward_model = reward_model
        self.sd_pipeline = sd_pipeline
        
        self.env = PureGRPOPromptEnvironment(qwen_model, reward_model, sd_pipeline, config)
        
        # 오직 정책 네트워크만! (Value Network 없음)
        self.policy = PureGRPOPolicy(qwen_model, config)
        
        # 참조 정책
        self.ref_policy = PureGRPOPolicy(qwen_model, config)
        self.ref_policy.load_state_dict(self.policy.state_dict())
        self.ref_policy.eval()
        
        # 오직 정책 파라미터만 학습
        trainable_params = list(self.policy.policy_head.parameters())
        self.optimizer = optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=0.01)
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-7)
        
        logger.info("🎯 순수 GRPO 트레이너 초기화 완료")
        logger.info(f"✅ Value Network 없음 - 오직 Policy Network만 사용")
        logger.info(f"📊 Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    def collect_rollouts(self, prompts: List[str]) -> List[Dict]:
        """롤아웃 수집 (Value 수집 없음)"""
        all_experiences = []
        
        for prompt_idx, user_prompt in enumerate(prompts):
            logger.info(f"Processing prompt {prompt_idx+1}/{len(prompts)}: '{user_prompt}'")
            
            for rollout_idx in range(self.config.num_rollouts):
                episode_experiences = []
                state = self.env.reset(user_prompt)
                done = False
                
                logger.info(f"  Rollout {rollout_idx+1}/{self.config.num_rollouts}")
                
                step_count = 0
                while not done and step_count < self.config.max_new_tokens:
                    # 정책에서 액션 선택 (Value 없음!)
                    action, log_prob, logits = self.policy.get_action_and_log_prob(state)
                    
                    # 참조 정책의 로그 확률
                    with torch.no_grad():
                        ref_logits = self.ref_policy(
                            state['input_ids'].unsqueeze(0),
                            state['attention_mask'].unsqueeze(0)
                        )
                        ref_log_prob = F.log_softmax(ref_logits, dim=-1)[0, action]
                    
                    next_state, reward, done, info = self.env.step(action)
                    
                    # Value 없는 경험 저장!
                    experience = {
                        'state': {k: v.clone() if torch.is_tensor(v) else v for k, v in state.items()},
                        'action': action,
                        'log_prob': log_prob,
                        'ref_log_prob': ref_log_prob,
                        'reward': reward,
                        'done': done,
                        'prompt_idx': prompt_idx,
                        'rollout_idx': rollout_idx,
                        'info': info
                        # ❌ 'value': value  # Value 없음!
                    }
                    
                    episode_experiences.append(experience)
                    state = next_state
                    step_count += 1
                
                all_experiences.extend(episode_experiences)
                
                if episode_experiences:
                    final_prompt = episode_experiences[-1]['info']['current_prompt']
                    final_reward = episode_experiences[-1]['reward']
                    logger.info(f"    Generated: '{final_prompt}' (reward: {final_reward:.3f})")
        
        return all_experiences
    
    def compute_grpo_advantages(self, experiences: List[Dict]) -> List[Dict]:
        """순수 GRPO Advantage 계산 (easyr1과 동일)"""
        # 프롬프트별 리워드 그룹화
        prompt_rewards = defaultdict(list)
        for exp in experiences:
            if exp['done']:
                prompt_rewards[exp['prompt_idx']].append(exp['reward'])
        
        # 그룹별 정규화 (easyr1과 동일한 방식)
        advantages = {}
        for prompt_idx, rewards in prompt_rewards.items():
            if len(rewards) > 1:
                mean_reward = np.mean(rewards)
                std_reward = np.std(rewards) + 1e-8
                normalized_rewards = [(r - mean_reward) / std_reward for r in rewards]
            else:
                normalized_rewards = [0.0]
            
            advantages[prompt_idx] = normalized_rewards
            logger.info(f"Prompt {prompt_idx} GRPO: rewards={rewards} -> advantages={normalized_rewards}")
        
        # 경험에 advantage 할당
        rollout_counters = defaultdict(int)
        for exp in experiences:
            if exp['done']:
                prompt_idx = exp['prompt_idx']
                rollout_idx = rollout_counters[prompt_idx]
                if prompt_idx in advantages and rollout_idx < len(advantages[prompt_idx]):
                    exp['advantage'] = advantages[prompt_idx][rollout_idx]
                else:
                    exp['advantage'] = 0.0
                rollout_counters[prompt_idx] += 1
            else:
                exp['advantage'] = 0.0
        
        return experiences
    
    def train_step(self, experiences: List[Dict]) -> Dict:
        """순수 GRPO 학습 스텝 (Value Loss 없음)"""
        if not experiences:
            return {}
        
        valid_experiences = [exp for exp in experiences if exp.get('advantage', 0) != 0]
        if not valid_experiences:
            logger.warning("No valid experiences for training")
            return {}
        
        batch_states = []
        actions = []
        old_log_probs = []
        ref_log_probs = []
        advantages = []
        
        for exp in valid_experiences:
            batch_states.append(exp['state'])
            actions.append(exp['action'])
            old_log_probs.append(exp['log_prob'])
            ref_log_probs.append(exp['ref_log_prob'])
            advantages.append(exp['advantage'])
            # ❌ values.append(exp['value'])  # Value 없음!
        
        if len(batch_states) == 0:
            return {}
        
        # 패딩을 위한 최대 길이 찾기
        max_length = max(state['input_ids'].size(0) for state in batch_states)
        
        # 패딩된 텐서 생성 및 GPU 0으로 이동
        padded_input_ids = []
        padded_attention_masks = []
        
        for state in batch_states:
            input_ids_tensor = state['input_ids']
            attention_mask_tensor = state['attention_mask']
            
            # 패딩 필요한 길이 계산
            pad_length = max_length - input_ids_tensor.size(0)
            
            if pad_length > 0:
                # 패딩 추가 (오른쪽에 패딩) - GPU 0에서
                padded_input = torch.cat([
                    input_ids_tensor,
                    torch.zeros(pad_length, dtype=input_ids_tensor.dtype, device="cuda:0")
                ])
                padded_mask = torch.cat([
                    attention_mask_tensor,
                    torch.zeros(pad_length, dtype=attention_mask_tensor.dtype, device="cuda:0")
                ])
            else:
                padded_input = input_ids_tensor.to("cuda:0")
                padded_mask = attention_mask_tensor.to("cuda:0")
            
            padded_input_ids.append(padded_input)
            padded_attention_masks.append(padded_mask)
        
        # 모든 텐서를 GPU 0으로 이동
        input_ids = torch.stack(padded_input_ids).to("cuda:0")
        attention_masks = torch.stack(padded_attention_masks).to("cuda:0")
        actions = torch.tensor(actions).to("cuda:0")
        old_log_probs = torch.stack(old_log_probs).to("cuda:0")
        ref_log_probs = torch.stack(ref_log_probs).to("cuda:0")
        advantages = torch.tensor(advantages, dtype=torch.float32).to("cuda:0")
        
        # 오직 정책 로짓만 계산! (Values 없음)
        policy_logits = self.policy(input_ids, attention_masks)
        
        new_log_probs = F.log_softmax(policy_logits, dim=-1)
        new_log_probs = new_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # PPO 정책 손실
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL 페널티
        kl_penalty = (new_log_probs - ref_log_probs).mean()
        
        # 엔트로피
        entropy = -(F.softmax(policy_logits, dim=-1) * F.log_softmax(policy_logits, dim=-1)).sum(-1).mean()
        
        # 순수 GRPO 총 손실 (Value Loss 없음!)
        total_loss = (policy_loss + 
                     self.config.kl_coef * kl_penalty - 
                     self.config.entropy_coef * entropy)
        
        # 역전파
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_penalty': kl_penalty.item(),
            'entropy': entropy.item(),
            'avg_advantage': advantages.mean().item(),
            'learning_rate': self.scheduler.get_last_lr()[0],
            'num_valid_experiences': len(valid_experiences)
        }
    
    def train(self, train_prompts: List[str], num_epochs: int = 10):
        """순수 GRPO 학습"""
        logger.info(f"🚀 순수 GRPO 학습 시작 (Value Network 없음)")
        logger.info(f"프롬프트: {len(train_prompts)}개, 에포크: {num_epochs}개")
        logger.info(f"Action Space: {self.env.vocab_size}개 토큰 (전체 어휘)")
        logger.info(f"✅ easyr1과 동일한 구조: 그룹 평균을 implicit baseline으로 사용")
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            logger.info("-" * 50)
            
            experiences = self.collect_rollouts(train_prompts)
            experiences = self.compute_grpo_advantages(experiences)
            metrics = self.train_step(experiences)
            
            logger.info(f"Epoch {epoch + 1} metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.6f}")
            
            if epoch % 2 == 0:
                self._log_sample_outputs(train_prompts[:2])
    
    def _log_sample_outputs(self, sample_prompts: List[str]):
        """샘플 출력 로깅"""
        logger.info("📝 Sample outputs:")
        for prompt in sample_prompts:
            state = self.env.reset(prompt)
            original_prompt = self.env.current_prompt
            
            # 몇 스텝 실행
            for _ in range(5):
                action, _, _ = self.policy.get_action_and_log_prob(state)
                state, _, done, info = self.env.step(action)
                if done:
                    break
            
            enhanced_prompt = self.env.current_prompt
            logger.info(f"  Original: {original_prompt}")
            logger.info(f"  Enhanced: {enhanced_prompt}")

def main():
    """테스트 실행"""
    logging.basicConfig(level=logging.INFO)
    
    class MockQwenModel:
        def __init__(self):
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            class MockModel:
                def __init__(self):
                    class Config:
                        hidden_size = 4096
                    self.config = Config()
                
                def __call__(self, **kwargs):
                    batch_size, seq_len = kwargs['input_ids'].shape
                    class Output:
                        last_hidden_state = torch.randn(batch_size, seq_len, 4096)
                    return Output()
            
            self.model = MockModel()
    
    class MockReward:
        def calculate_reward(self, original, enhanced, image):
            return np.random.uniform(5.0, 9.0)
    
    class MockSD:
        def __call__(self, **kwargs):
            from PIL import Image
            class Result:
                images = [Image.new('RGB', (512, 512), color='red')]
            return Result()
    
    config = PureGRPOConfig(
        learning_rate=1e-6,
        batch_size=2,
        num_rollouts=3,
        max_new_tokens=10,
        top_k=50
    )
    
    qwen = MockQwenModel()
    reward = MockReward()
    sd = MockSD()
    
    trainer = PureGRPOTrainer(qwen, reward, sd, config)
    
    test_prompts = ["a cat sitting", "beautiful sunset"]
    trainer.train(test_prompts, num_epochs=2)

if __name__ == "__main__":
    main() 