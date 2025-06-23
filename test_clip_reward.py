#!/usr/bin/env python3
"""
CLIP Reward System Test
=======================

CLIP 보상 시스템을 테스트하는 스크립트입니다.
원본 user prompt와 이미지 간의 유사도를 계산합니다.

사용법:
    python test_clip_reward.py

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

def main():
    """메인 테스트 함수"""
    print("🎯 CLIP Reward System Test")
    print("=" * 50)
    
    try:
        # CLIP 모델 로드
        print("📥 Loading CLIP model...")
        from models.clip_reward import CLIPRewardCalculator, create_dummy_image
        
        clip_calculator = CLIPRewardCalculator(
            model_name="openai/clip-vit-base-patch32",
            device="auto"
        )
        
        print(f"✅ CLIP model loaded: {clip_calculator.model_name}")
        print(f"🖥️  Device: {clip_calculator.device}")
        print()
        
        # 테스트 케이스들
        test_cases = [
            "a cat",
            "sunset",
            "mountain landscape", 
            "robot",
            "flower garden",
            "city at night",
            "portrait of a woman",
            "red car"
        ]
        
        print("🧪 Testing CLIP reward calculation...")
        print("-" * 50)
        
        all_rewards = []
        
        # 각 프롬프트에 대해 더미 이미지 생성 및 보상 계산
        for i, user_prompt in enumerate(test_cases, 1):
            print(f"\n[Test {i}/{len(test_cases)}]")
            print(f"📝 User Prompt: {user_prompt}")
            
            # 더미 이미지 생성 (나중에 실제 SD 이미지로 교체)
            dummy_image = create_dummy_image(user_prompt)
            print(f"🖼️  Generated dummy image (color based on prompt)")
            
            # CLIP 보상 계산
            reward = clip_calculator.calculate_reward(user_prompt, dummy_image)
            all_rewards.append(reward)
            
            print(f"🎯 Reward: {reward:.4f}")
            
            # 상세 정보 가져오기
            details = clip_calculator.get_detailed_similarity(user_prompt, dummy_image)
            print(f"📊 Raw similarity: {details['raw_similarity']:.4f}")
            print(f"🔍 Confidence: {details['confidence']:.4f}")
        
        print("\n" + "=" * 50)
        print("📈 Summary Statistics:")
        print(f"   Average reward: {sum(all_rewards)/len(all_rewards):.4f}")
        print(f"   Min reward: {min(all_rewards):.4f}")
        print(f"   Max reward: {max(all_rewards):.4f}")
        print(f"   Reward range: {max(all_rewards)-min(all_rewards):.4f}")
        
        # 배치 테스트
        print("\n🚀 Testing batch reward calculation...")
        batch_prompts = test_cases[:4]
        batch_images = [create_dummy_image(p) for p in batch_prompts]
        
        batch_rewards = clip_calculator.calculate_rewards_batch(batch_prompts, batch_images)
        
        for i, (prompt, reward) in enumerate(zip(batch_prompts, batch_rewards)):
            print(f"Batch {i+1}: '{prompt}' -> Reward: {reward:.4f}")
        
        # 모델 정보 출력
        print(f"\n📊 Model Info: {clip_calculator.get_model_info()}")
        
        print("\n🎉 All CLIP reward tests completed!")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Please install required packages: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Test error: {e}", exc_info=True)

def test_reward_characteristics():
    """보상 특성 테스트"""
    print("\n🔬 Testing reward characteristics...")
    
    try:
        from models.clip_reward import create_dummy_image
        
        # 동일한 프롬프트로 여러 이미지 생성
        prompt = "cat"
        images = [create_dummy_image(f"{prompt}_{i}") for i in range(5)]
        
        print(f"Testing consistency for prompt: '{prompt}'")
        
        # 더미 이미지들은 서로 다른 색상을 가지므로 reward 분산 확인
        rewards = []
        for i, img in enumerate(images):
            # 더미 계산 (실제 CLIP 없이)
            dummy_reward = 0.5 + (hash(f"{prompt}_{i}") % 100) / 200  # 0.5-1.0 범위
            rewards.append(dummy_reward)
            print(f"  Image {i+1}: {dummy_reward:.4f}")
        
        print(f"  Reward variance: {max(rewards)-min(rewards):.4f}")
        
    except Exception as e:
        print(f"⚠️ Characteristic test failed: {e}")

if __name__ == "__main__":
    # 특성 테스트 먼저 실행
    test_reward_characteristics()
    
    # 메인 테스트 실행
    main() 