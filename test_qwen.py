#!/usr/bin/env python3
"""
QWEN 7B Prompt Enhancement Test
===============================

QWEN 7B 모델의 프롬프트 향상 기능을 테스트하는 스크립트입니다.

사용법:
    python test_qwen.py

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
    print("🚀 QWEN 7B Prompt Enhancement Test")
    print("=" * 50)
    
    try:
        # QWEN 모델 로드
        print("📥 Loading QWEN model...")
        from models.qwen_wrapper import QwenWrapper
        
        qwen = QwenWrapper(
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            device="auto",
            max_new_tokens=100,
            temperature=0.7
        )
        
        print(f"✅ Model loaded: {qwen.model_name}")
        print(f"🖥️  Device: {qwen.device}")
        print()
        
        # 테스트 프롬프트들
        test_prompts = [
            "a cat",
            "sunset",
            "mountain landscape",
            "robot",
            "flower garden",
            "city at night",
            "portrait of a woman",
            "abstract art"
        ]
        
        print("🧪 Testing prompt enhancement...")
        print("-" * 50)
        
        # 각 프롬프트 테스트
        for i, user_prompt in enumerate(test_prompts, 1):
            print(f"\n[Test {i}/{len(test_prompts)}]")
            print(f"📝 Original: {user_prompt}")
            
            # 프롬프트 향상
            result = qwen.enhance_prompt(user_prompt)
            
            print(f"➕ With Placeholder: {result['prompt_with_placeholder']}")
            print(f"✨ Enhanced: {result['enhanced_prompt']}")
            print(f"📏 Length: {len(result['original_prompt'])} -> {len(result['prompt_with_placeholder'])} -> {len(result['enhanced_prompt'])}")
            
            # 품질 체크
            quality_score = evaluate_enhancement_quality(
                result['original_prompt'], 
                result['enhanced_prompt']
            )
            print(f"🎯 Quality Score: {quality_score:.2f}/5.0")
            
            if result['raw_output'] != result['enhanced_prompt']:
                print(f"🔧 Raw output: {result['raw_output'][:100]}...")
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed!")
        
        # 배치 테스트
        print("\n🚀 Testing batch enhancement...")
        batch_results = qwen.enhance_prompts_batch(test_prompts[:3])
        
        for i, result in enumerate(batch_results):
            print(f"Batch {i+1}: '{result['original_prompt']}' -> '{result['enhanced_prompt'][:50]}...'")
        
        # 모델 정보 출력
        print(f"\n📊 Model Info: {qwen.get_model_info()}")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Please install required packages: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Test error: {e}", exc_info=True)

def evaluate_enhancement_quality(original: str, enhanced: str) -> float:
    """
    프롬프트 향상 품질 평가
    
    평가 기준:
    1. 길이 증가 (더 상세함)
    2. 원본 키워드 포함
    3. 새로운 디테일 추가
    4. 문법 및 구조
    """
    score = 0.0
    
    # 1. 길이 증가 (1점)
    if len(enhanced) > len(original):
        length_ratio = len(enhanced) / len(original)
        score += min(1.0, length_ratio / 3.0)  # 3배 이상 길어지면 만점
    
    # 2. 원본 키워드 포함 (1점)
    original_words = set(original.lower().split())
    enhanced_words = set(enhanced.lower().split())
    
    if original_words.issubset(enhanced_words):
        score += 1.0
    else:
        # 부분 점수
        overlap = len(original_words.intersection(enhanced_words))
        score += overlap / len(original_words)
    
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

def quick_test():
    """빠른 테스트 함수 (모델 로드 없이)"""
    print("🏃‍♂️ Quick Test (without model loading)")
    
    # 더미 결과로 품질 평가 테스트
    test_cases = [
        ("cat", "a beautiful fluffy cat sitting gracefully, high quality, detailed fur, professional photography"),
        ("sunset", "sunset"),  # 향상되지 않은 경우
        ("robot", "futuristic robot with metallic finish, cinematic lighting, 4k resolution, sci-fi style")
    ]
    
    for original, enhanced in test_cases:
        score = evaluate_enhancement_quality(original, enhanced)
        print(f"'{original}' -> '{enhanced}' | Score: {score:.2f}/5.0")

if __name__ == "__main__":
    # 빠른 테스트 먼저 실행
    quick_test()
    print()
    
    # 메인 테스트 실행
    main() 