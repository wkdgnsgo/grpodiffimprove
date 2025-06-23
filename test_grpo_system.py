"""
GRPO System Test Script
======================

전체 GRPO 시스템을 테스트하는 스크립트입니다.
실제 학습 전에 모든 컴포넌트가 제대로 작동하는지 확인합니다.

사용법:
    python test_grpo_system.py

Author: AI Assistant
Date: 2025-01-22
"""

import sys
import os
import logging
from typing import List, Dict, Any

# 로컬 모듈 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_qwen_model():
    """QWEN 모델 테스트"""
    logger.info("🔍 Testing QWEN model...")
    
    try:
        from models.qwen_wrapper import QwenWrapper
        
        # 모델 초기화
        qwen = QwenWrapper()
        
        # 테스트 프롬프트
        test_prompt = "a cute cat"
        
        # 프롬프트 향상 테스트
        result = qwen.enhance_prompt(test_prompt)
        
        logger.info(f"✅ QWEN test passed")
        logger.info(f"  Original: {result['original_prompt']}")
        logger.info(f"  With placeholder: {result['prompt_with_placeholder'][:50]}...")
        logger.info(f"  Enhanced: {result['enhanced_prompt'][:50]}...")
        
        return True, qwen
        
    except Exception as e:
        logger.error(f"❌ QWEN test failed: {e}")
        return False, None

def test_sd3_generator():
    """SD3 생성기 테스트"""
    logger.info("🔍 Testing SD3 generator...")
    
    try:
        from models.sd3_generator import SD3Generator
        
        # 작은 이미지 크기로 테스트
        generator = SD3Generator(
            height=256,
            width=256,
            num_inference_steps=5  # 빠른 테스트
        )
        
        # 더미 이미지 생성 테스트
        test_prompt = "a cute cat, high quality, detailed"
        image = generator.generate_image(test_prompt)
        
        if image is not None:
            logger.info(f"✅ SD3 test passed - Generated image size: {image.size}")
            return True, generator
        else:
            logger.error("❌ SD3 test failed - No image generated")
            return False, None
        
    except Exception as e:
        logger.error(f"❌ SD3 test failed: {e}")
        return False, None

def test_clip_reward():
    """CLIP 보상 계산기 테스트"""
    logger.info("🔍 Testing CLIP reward calculator...")
    
    try:
        from models.clip_reward import CLIPRewardCalculator
        from PIL import Image
        import numpy as np
        
        # 보상 계산기 초기화
        clip_calc = CLIPRewardCalculator()
        
        # 더미 이미지 생성 (RGB 이미지)
        dummy_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # 테스트 프롬프트
        test_prompt = "a cute cat"
        
        # 보상 계산 테스트
        reward = clip_calc.calculate_reward(test_prompt, dummy_image)
        
        logger.info(f"✅ CLIP test passed - Reward: {reward:.4f}")
        return True, clip_calc
        
    except Exception as e:
        logger.error(f"❌ CLIP test failed: {e}")
        return False, None

def test_environment():
    """환경 테스트"""
    logger.info("🔍 Testing GRPO environment...")
    
    try:
        from training.grpo_trainer import PromptEnvironment, GRPOConfig
        
        # 이전 테스트 결과 가져오기
        qwen_success, qwen_model = test_qwen_model()
        if not qwen_success:
            return False
        
        # 더미 생성기와 보상 계산기
        class DummySD3Generator:
            def generate_image(self, prompt):
                from PIL import Image
                return Image.new('RGB', (224, 224), color=(100, 150, 200))
        
        class DummyCLIPCalculator:
            def calculate_reward(self, prompt, image):
                return 0.75  # 더미 보상
        
        # 환경 설정
        config = GRPOConfig(max_new_tokens=3)
        env = PromptEnvironment(
            qwen_model, 
            DummySD3Generator(), 
            DummyCLIPCalculator(), 
            config
        )
        
        # 환경 테스트
        test_prompt = "a cute cat"
        state = env.reset(test_prompt)
        
        logger.info(f"  Initial state shape: {state.shape}")
        logger.info(f"  Action space size: {env.get_action_space_size()}")
        
        # 몇 개 액션 실행
        for step in range(3):
            action = step % env.get_action_space_size()  # 간단한 액션 선택
            next_state, reward, done = env.step(action)
            logger.info(f"  Step {step}: action={action}, reward={reward:.3f}, done={done}")
            
            if done:
                break
        
        logger.info("✅ Environment test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment test failed: {e}")
        return False

def test_grpo_trainer():
    """GRPO 트레이너 테스트"""
    logger.info("🔍 Testing GRPO trainer...")
    
    try:
        from training.grpo_trainer import GRPOTrainer, GRPOConfig
        
        # 이전 테스트 결과 가져오기
        qwen_success, qwen_model = test_qwen_model()
        if not qwen_success:
            return False
        
        # 더미 생성기와 보상 계산기
        class DummySD3Generator:
            def generate_image(self, prompt):
                from PIL import Image
                return Image.new('RGB', (224, 224), color=(100, 150, 200))
        
        class DummyCLIPCalculator:
            def calculate_reward(self, prompt, image):
                import random
                return random.uniform(0.5, 0.9)  # 랜덤 보상
        
        # 트레이너 설정
        config = GRPOConfig(
            learning_rate=1e-4,
            group_size=2,
            num_iterations=2,
            grpo_epochs=2,
            max_new_tokens=3
        )
        
        trainer = GRPOTrainer(
            qwen_model, 
            DummySD3Generator(), 
            DummyCLIPCalculator(), 
            config
        )
        
        # Challenging 프롬프트로 테스트
        test_prompts = [
            "a cute cat", 
            "a purple rabbit eating carrots",  # Challenging case
            "a transparent glass butterfly"     # Very challenging case
        ]
        
        for i in range(2):
            logger.info(f"  Testing iteration {i+1}...")
            results = trainer.train_iteration(test_prompts)
            
            logger.info(f"    Avg reward: {results['avg_reward']:.4f}")
            logger.info(f"    Avg length: {results['avg_length']:.1f}")
            logger.info(f"    Policy obj: {results['policy_objective']:.6f}")
            logger.info(f"    KL div: {results['kl_divergence']:.6f}")
            logger.info(f"    Entropy: {results['entropy']:.4f}")
        
        logger.info("✅ GRPO trainer test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ GRPO trainer test failed: {e}")
        return False

def main():
    """메인 테스트 함수"""
    logger.info("🚀 Starting GRPO System Tests")
    logger.info("=" * 50)
    
    tests = [
        ("QWEN Model", test_qwen_model),
        ("SD3 Generator", test_sd3_generator),
        ("CLIP Reward", test_clip_reward),
        ("Environment", test_environment),
        ("GRPO Trainer", test_grpo_trainer)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            if test_name in ["QWEN Model", "SD3 Generator", "CLIP Reward"]:
                success, component = test_func()
                results[test_name] = (success, component)
            else:
                success = test_func()
                results[test_name] = (success, None)
                
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results[test_name] = (False, None)
    
    # 결과 요약
    logger.info(f"\n{'='*50}")
    logger.info("📊 Test Results Summary")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(tests)
    
    for test_name, (success, _) in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready for training.")
        return True
    else:
        logger.info("⚠️ Some tests failed. Please fix issues before training.")
        return False

if __name__ == "__main__":
    main() 