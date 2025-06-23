#!/usr/bin/env python3
"""
Integrated System Test: QWEN VL + CLIP Reward
==============================================

QWEN VL 모델과 CLIP 보상 시스템을 통합하여 테스트하는 스크립트입니다.

전체 플로우:
User Prompt → QWEN VL → Enhanced Prompt → (Dummy Image) → CLIP Reward(User Prompt vs Image)

사용법:
    python test_integrated_system.py

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
    """메인 통합 테스트 함수"""
    print("🚀 Integrated System Test: QWEN VL + CLIP Reward")
    print("=" * 60)
    
    try:
        # 1. QWEN VL 모델 로드
        print("📥 Loading QWEN VL model...")
        from models.qwen_wrapper import QwenWrapper
        
        qwen = QwenWrapper(
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            device="auto",
            max_new_tokens=100,
            temperature=0.7
        )
        
        print(f"✅ QWEN VL loaded: {qwen.model_name}")
        print(f"🖥️  Device: {qwen.device}")
        
        # 2. CLIP Reward 모델 로드
        print("\n📥 Loading CLIP Reward model...")
        from models.clip_reward import CLIPRewardCalculator, create_dummy_image
        
        clip_calculator = CLIPRewardCalculator(
            model_name="openai/clip-vit-base-patch32",
            device="auto"
        )
        
        print(f"✅ CLIP loaded: {clip_calculator.model_name}")
        print(f"🖥️  Device: {clip_calculator.device}")
        print()
        
        # 3. 테스트 케이스들
        test_cases = [
            "a cat",
            "sunset",
            "robot",
            "flower garden",
            "city at night"
        ]
        
        print("🧪 Testing integrated pipeline...")
        print("-" * 60)
        
        all_results = []
        
        # 각 테스트 케이스 실행
        for i, user_prompt in enumerate(test_cases, 1):
            print(f"\n[Pipeline Test {i}/{len(test_cases)}]")
            print(f"📝 User Prompt: '{user_prompt}'")
            
            # Step 1: QWEN VL로 프롬프트 향상
            enhancement_result = qwen.enhance_prompt(user_prompt)
            enhanced_prompt = enhancement_result['enhanced_prompt']
            
            print(f"✨ Enhanced: '{enhanced_prompt[:80]}...'")
            print(f"📏 Length: {len(user_prompt)} -> {len(enhanced_prompt)}")
            
            # Step 2: 더미 이미지 생성 (나중에 실제 SD로 교체)
            # 주의: Enhanced prompt로 이미지를 생성하지만, 
            # CLIP 보상은 Original user prompt로 계산!
            dummy_image = create_dummy_image(enhanced_prompt)
            print(f"🖼️  Generated dummy image from enhanced prompt")
            
            # Step 3: CLIP 보상 계산 (원본 user prompt 사용!)
            reward = clip_calculator.calculate_reward(user_prompt, dummy_image)
            
            print(f"🎯 CLIP Reward: {reward:.4f} (user prompt vs image)")
            
            # Step 4: 상세 분석
            quality_score = evaluate_enhancement_quality(user_prompt, enhanced_prompt)
            
            result = {
                'user_prompt': user_prompt,
                'enhanced_prompt': enhanced_prompt,
                'enhancement_quality': quality_score,
                'clip_reward': reward,
                'prompt_improvement': len(enhanced_prompt) / len(user_prompt)
            }
            
            all_results.append(result)
            
            print(f"📊 Enhancement Quality: {quality_score:.2f}/5.0")
            print(f"📈 Length Ratio: {result['prompt_improvement']:.2f}x")
        
        print("\n" + "=" * 60)
        print("📈 Final Results Summary:")
        print("-" * 60)
        
        # 통계 계산
        avg_reward = sum(r['clip_reward'] for r in all_results) / len(all_results)
        avg_quality = sum(r['enhancement_quality'] for r in all_results) / len(all_results)
        avg_improvement = sum(r['prompt_improvement'] for r in all_results) / len(all_results)
        
        print(f"🎯 Average CLIP Reward: {avg_reward:.4f}")
        print(f"📊 Average Enhancement Quality: {avg_quality:.2f}/5.0")
        print(f"📈 Average Length Improvement: {avg_improvement:.2f}x")
        
        # 상관관계 분석
        print(f"\n🔍 Analysis:")
        high_quality_results = [r for r in all_results if r['enhancement_quality'] >= 4.0]
        if high_quality_results:
            high_quality_reward = sum(r['clip_reward'] for r in high_quality_results) / len(high_quality_results)
            print(f"   High quality enhancements (≥4.0) average reward: {high_quality_reward:.4f}")
        
        # 개별 결과 표시
        print(f"\n📋 Individual Results:")
        for i, result in enumerate(all_results, 1):
            print(f"{i}. '{result['user_prompt']}' -> Quality: {result['enhancement_quality']:.2f}, Reward: {result['clip_reward']:.4f}")
        
        print("\n🎉 Integrated system test completed successfully!")
        
        # 시스템 요약
        print(f"\n📊 System Configuration:")
        print(f"   QWEN VL: {qwen.get_model_info()}")
        print(f"   CLIP: {clip_calculator.get_model_info()}")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Please install required packages: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Integrated test failed: {e}")
        logger.error(f"Test error: {e}", exc_info=True)

def evaluate_enhancement_quality(original: str, enhanced: str) -> float:
    """
    프롬프트 향상 품질 평가 (기존 함수 재사용)
    """
    score = 0.0
    
    # 1. 길이 증가 (1점)
    if len(enhanced) > len(original):
        length_ratio = len(enhanced) / len(original)
        score += min(1.0, length_ratio / 3.0)
    
    # 2. 원본 키워드 포함 (1점)
    original_words = set(original.lower().split())
    enhanced_words = set(enhanced.lower().split())
    
    if original_words.issubset(enhanced_words):
        score += 1.0
    else:
        overlap = len(original_words.intersection(enhanced_words))
        score += overlap / len(original_words) if original_words else 0
    
    # 3. 새로운 디테일 추가 (1점)
    new_words = enhanced_words - original_words
    if len(new_words) >= 5:
        score += 1.0
    else:
        score += len(new_words) / 5.0
    
    # 4. 품질 키워드 포함 (1점)
    quality_keywords = [
        'high quality', 'detailed', 'professional', 'cinematic', 'artistic',
        'beautiful', 'stunning', 'masterpiece', '4k', '8k', 'ultra',
        'realistic', 'photorealistic', 'style', 'lighting'
    ]
    
    found_quality_words = sum(1 for kw in quality_keywords if kw in enhanced.lower())
    score += min(1.0, found_quality_words / 3.0)
    
    # 5. 문장 구조 (1점)
    if enhanced != original and len(enhanced.split()) >= 5:
        score += 1.0
    
    return score

def quick_integration_test():
    """빠른 통합 테스트 (모델 로드 없이)"""
    print("🏃‍♂️ Quick Integration Test (without model loading)")
    print("-" * 50)
    
    # 더미 데이터로 파이프라인 테스트
    test_cases = [
        ("cat", "a beautiful fluffy cat sitting gracefully, high quality, detailed fur, professional photography"),
        ("sunset", "stunning sunset over the ocean, golden hour lighting, cinematic composition, 4k resolution"),
        ("robot", "futuristic robot with metallic finish, sci-fi environment, detailed mechanical parts")
    ]
    
    for user_prompt, enhanced_prompt in test_cases:
        print(f"\n📝 User: '{user_prompt}'")
        print(f"✨ Enhanced: '{enhanced_prompt[:60]}...'")
        
        # 더미 보상 계산
        dummy_reward = 0.7 + (hash(user_prompt) % 100) / 500  # 0.7-0.9 범위
        quality = evaluate_enhancement_quality(user_prompt, enhanced_prompt)
        
        print(f"🎯 Dummy Reward: {dummy_reward:.4f}")
        print(f"📊 Quality: {quality:.2f}/5.0")

if __name__ == "__main__":
    # 빠른 테스트 먼저 실행
    quick_integration_test()
    print("\n" + "=" * 60)
    
    # 메인 통합 테스트 실행
    main() 