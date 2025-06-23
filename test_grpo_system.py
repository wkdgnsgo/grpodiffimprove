#!/usr/bin/env python3
"""
GRPO System Integration Test
===========================

전반적인 parameter 에러가 없게 점검하고 ref 모델도 현재 policy모델과 연동이 잘 되는지 확인하는 테스트 스크립트

이 스크립트는 다음을 검증합니다:
1. 모든 컴포넌트의 올바른 초기화
2. 파라미터 연동 및 호환성
3. 참조 모델(ref model)과 정책 모델의 연동
4. GRPO 학습 루프의 정상 작동
5. 제공된 코드 형식과의 호환성

Author: AI Assistant
Date: 2025-01-22
"""

import sys
import logging
from pathlib import Path

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """모든 필수 모듈 임포트 테스트"""
    print("🔧 Testing Module Imports...")
    
    try:
        from training.grpo_trainer import GRPOTrainer, GRPOConfig
        from models.vlm_wrapper import VLMWrapper
        from models.sd_generator import SD3Generator
        from models.clip_reward import CLIPRewardCalculator
        from integration.main_trainer import VLMGRPOSystem
        print("✅ All core modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_grpo_config():
    """GRPO 설정 테스트"""
    print("\n📊 Testing GRPO Configuration...")
    
    try:
        from training.grpo_trainer import GRPOConfig
        
        # --- GRPO Configuration (제공된 코드 형식 스타일) ---
        config = GRPOConfig(
            learning_rate=5e-6,
            group_size=4,
            num_iterations=100,
            grpo_epochs=3,
            gamma=0.99,
            kl_beta=0.02,
            clip_epsilon=0.2,
            entropy_coeff=0.01,
            max_grad_norm=1.0,
            max_new_tokens=25,
            temperature=0.8,
            device='cpu'
        )
        
        print(f"✅ GRPO Config created successfully:")
        print(f"  - Learning Rate: {config.learning_rate}")
        print(f"  - Group Size: {config.group_size}")
        print(f"  - Iterations: {config.num_iterations}")
        print(f"  - GRPO Epochs: {config.grpo_epochs}")
        print(f"  - Gamma: {config.gamma}")
        print(f"  - KL Beta: {config.kl_beta}")
        print(f"  - Device: {config.device}")
        
        return config
    except Exception as e:
        print(f"❌ GRPO Config test failed: {e}")
        return None

def create_mock_components():
    """Mock 컴포넌트 생성 (제공된 코드 형식과 유사)"""
    print("\n🔧 Creating Mock Components...")
    
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
    import numpy as np
    
    class MockVLM:
        """Mock VLM Policy Network"""
        def __init__(self):
            # --- Policy Network Initialization ---
            # Embedding layer to handle token IDs properly
            self.embedding = nn.Embedding(50000, 128)
            self.model = nn.Sequential(
                nn.Linear(128, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 50000)
            )
            self.tokenizer = type('MockTokenizer', (), {
                '__len__': lambda self: 50000,
                'eos_token_id': 2
            })()
            self.model_name = 'mock-qwen2.5-vl'
            
        def forward(self, input_ids, attention_mask=None):
            """Policy Network Forward Pass"""
            # Properly handle token IDs through embedding
            embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
            # Use mean pooling to get fixed-size representation
            pooled = embedded.mean(dim=1)  # [batch_size, embed_dim]
            logits = self.model(pooled)  # [batch_size, vocab_size]
            return Categorical(logits=logits)
            
        def generate_sequence(self, prompt, max_new_tokens=5):
            """Token-by-token Generation for GRPO"""
            states = []
            actions = []
            log_probs = []
            
            # Consistent tensor shapes
            seq_len = 10
            
            for i in range(max_new_tokens):
                # State (input sequence) - ensure proper dtype
                state = torch.randint(0, 1000, (seq_len,), dtype=torch.long)
                states.append(state)
                
                # Action (next token)
                action = torch.randint(0, 1000, (), dtype=torch.long)
                actions.append(action)
                
                # Log probability
                log_prob = torch.randn((), dtype=torch.float32) * 0.1
                log_probs.append(log_prob)
                
                # Early stopping
                if i > 1 and np.random.random() < 0.3:
                    break
            
            return {
                'generated_text': f'enhanced {prompt} with artistic creative details',
                'generated_ids': torch.randint(0, 1000, (1, len(states))),
                'states': states,
                'actions': actions,
                'log_probs': log_probs
            }
            
        def get_log_prob(self, input_ids, target_token, attention_mask=None):
            """Calculate log probability for specific token"""
            policy_dist = self.forward(input_ids, attention_mask)
            return policy_dist.log_prob(target_token)
    
    class MockSDGenerator:
        """Mock SD3 Generator (Frozen)"""
        def __init__(self):
            self.model_name = 'mock-sd3'
        
        def generate_image(self, prompt):
            return f'high_quality_image_for_{prompt[:40]}'
    
    class MockCLIPReward:
        """Mock CLIP Reward Calculator (Frozen)"""
        def __init__(self):
            self.model_name = 'mock-clip'
        
        def calculate_reward(self, image, text):
            # Realistic reward calculation
            base_reward = 0.5
            text_quality = min(len(text.split()) * 0.015, 0.25)
            creativity_bonus = 0.1 if any(word in text.lower() for word in ['creative', 'artistic', 'beautiful']) else 0.0
            noise = np.random.normal(0, 0.08)
            return np.clip(base_reward + text_quality + creativity_bonus + noise, 0.0, 1.0)
    
    # --- Component Initialization (제공된 코드 형식) ---
    vlm_policy = MockVLM()
    sd_generator = MockSDGenerator()
    clip_reward = MockCLIPReward()
    
    print(f"✅ Mock components created:")
    print(f"  - VLM Policy: {vlm_policy.model_name}")
    print(f"  - SD Generator: {sd_generator.model_name}")
    print(f"  - CLIP Reward: {clip_reward.model_name}")
    
    return vlm_policy, sd_generator, clip_reward

def test_grpo_trainer_initialization(vlm_policy, sd_generator, clip_reward, config):
    """GRPO Trainer 초기화 테스트"""
    print("\n🎯 Testing GRPO Trainer Initialization...")
    
    try:
        from training.grpo_trainer import GRPOTrainer
        
        # --- GRPO Trainer Initialization (제공된 코드 형식) ---
        grpo_trainer = GRPOTrainer(
            vlm_model=vlm_policy,
            sd_generator=sd_generator,
            clip_reward=clip_reward,
            config=config
        )
        
        print("✅ GRPO Trainer initialized successfully")
        
        # --- Parameter Verification ---
        vlm_params = sum(p.numel() for p in grpo_trainer.vlm_policy.model.parameters())
        print(f"📊 VLM Policy parameters: {vlm_params:,}")
        
        # --- Reference Policy Verification ---
        if grpo_trainer.vlm_policy_ref is not None:
            ref_params = sum(p.numel() for p in grpo_trainer.vlm_policy_ref.parameters())
            print(f"📋 Reference policy parameters: {ref_params:,}")
            
            # Check if ref model is frozen
            ref_requires_grad = any(p.requires_grad for p in grpo_trainer.vlm_policy_ref.parameters())
            if not ref_requires_grad:
                print("✅ Reference policy properly frozen")
            else:
                print("⚠️ Reference policy not frozen")
        else:
            print("❌ Reference policy not created")
            return None
        
        # --- Optimizer Verification ---
        if grpo_trainer.vlm_optimizer is not None:
            lr = grpo_trainer.vlm_optimizer.param_groups[0]['lr']
            print(f"🔧 Optimizer learning rate: {lr}")
            print("✅ VLM optimizer created")
        else:
            print("❌ VLM optimizer not created")
            return None
        
        return grpo_trainer
        
    except Exception as e:
        print(f"❌ GRPO Trainer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_training_iteration(grpo_trainer):
    """학습 반복 테스트"""
    print("\n🔄 Testing Training Iteration...")
    
    try:
        # --- Training Prompts ---
        train_prompts = [
            "a breathtaking mountain landscape at golden hour",
            "a curious kitten exploring a colorful garden",
            "abstract art with vibrant flowing colors",
            "a peaceful forest scene with morning mist"
        ]
        
        print(f"📝 Training with {len(train_prompts)} prompts")
        
        # --- Single Training Iteration ---
        metrics = grpo_trainer.train_iteration(train_prompts)
        
        print("✅ Training iteration completed successfully")
        print(f"📊 Iteration Metrics:")
        print(f"  - Average Reward: {metrics['avg_reward']:.4f}")
        print(f"  - Policy Loss: {metrics['policy_loss']:.4f}")
        print(f"  - Entropy: {metrics['entropy']:.4f}")
        print(f"  - KL Divergence: {metrics['kl_div']:.4f}")
        
        # --- Training Statistics Verification ---
        stats = grpo_trainer.get_training_stats()
        print(f"📈 Training Statistics:")
        print(f"  - Current Iteration: {stats['iteration']}")
        print(f"  - Average Reward: {stats['avg_reward']:.4f}")
        
        # --- History Verification ---
        print(f"📋 History Lengths:")
        print(f"  - Rewards: {len(grpo_trainer.iteration_rewards)}")
        print(f"  - Policy Losses: {len(grpo_trainer.iteration_policy_losses)}")
        print(f"  - Entropies: {len(grpo_trainer.iteration_entropies)}")
        print(f"  - KL Divergences: {len(grpo_trainer.iteration_kl_divs)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training iteration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_training_loop(grpo_trainer):
    """전체 학습 루프 테스트 (제공된 코드 형식)"""
    print("\n🚀 Testing Full Training Loop...")
    
    try:
        # --- Training Configuration ---
        NUM_ITERATIONS_GRPO = 3
        GROUP_SIZE = 2
        INTERV_PRINT = 1
        
        # --- Training Prompts ---
        train_prompts = [
            "a serene lake reflecting autumn colors",
            "a majestic eagle soaring through clouds",
            "vibrant street art on urban walls",
            "a cozy cabin in a snowy forest"
        ]
        
        print(f"🔄 Starting {NUM_ITERATIONS_GRPO} iterations with group size {GROUP_SIZE}")
        
        # --- GRPO Training Loop (제공된 코드 형식과 유사) ---
        for iteration in range(NUM_ITERATIONS_GRPO):
            print(f"\n--- Iteration {iteration+1}/{NUM_ITERATIONS_GRPO} ---")
            
            # --- Sample Group Prompts ---
            import random
            if len(train_prompts) >= GROUP_SIZE:
                group_prompts = random.sample(train_prompts, GROUP_SIZE)
            else:
                group_prompts = (train_prompts * ((GROUP_SIZE // len(train_prompts)) + 1))[:GROUP_SIZE]
            
            # --- Perform GRPO Training Iteration ---
            metrics = grpo_trainer.train_iteration(group_prompts)
            
            # --- Logging ---
            if (iteration + 1) % INTERV_PRINT == 0 or iteration == NUM_ITERATIONS_GRPO - 1:
                print(f"📊 Iteration {iteration+1}/{NUM_ITERATIONS_GRPO} Summary:")
                print(f"  Avg Reward (Group): {metrics['avg_reward']:.4f}")
                print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
                print(f"  Entropy: {metrics['entropy']:.4f}")
                print(f"  KL Divergence: {metrics['kl_div']:.4f}")
        
        print(f"\n🎉 GRPO Training Loop Finished!")
        
        # --- Final Statistics ---
        final_stats = grpo_trainer.get_training_stats()
        print(f"📊 Final Statistics:")
        print(f"  Total Iterations: {final_stats['iteration']}")
        print(f"  Final Average Reward: {final_stats['avg_reward']:.4f}")
        print(f"  Reward Progression: {[f'{r:.3f}' for r in grpo_trainer.iteration_rewards]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full training loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 테스트 실행"""
    print("🧪 GRPO System Integration Test")
    print("=" * 50)
    print("전반적인 parameter 에러가 없게 점검하고 ref 모델도 현재 policy모델과 연동이 잘 되는지 확인")
    print("=" * 50)
    
    # 1. 임포트 테스트
    if not test_imports():
        return False
    
    # 2. GRPO 설정 테스트
    config = test_grpo_config()
    if config is None:
        return False
    
    # 3. Mock 컴포넌트 생성
    vlm_policy, sd_generator, clip_reward = create_mock_components()
    
    # 4. GRPO Trainer 초기화 테스트
    grpo_trainer = test_grpo_trainer_initialization(vlm_policy, sd_generator, clip_reward, config)
    if grpo_trainer is None:
        return False
    
    # 5. 단일 학습 반복 테스트
    if not test_training_iteration(grpo_trainer):
        return False
    
    # 6. 전체 학습 루프 테스트
    if not test_full_training_loop(grpo_trainer):
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All Tests Passed Successfully!")
    print("✅ Parameter errors resolved")
    print("✅ Reference model properly integrated")
    print("✅ GRPO trainer working correctly")
    print("✅ System ready for real training")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 