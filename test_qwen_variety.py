#!/usr/bin/env python3
"""
QWEN VL 프롬프트 다양성 테스트
placeholder_template 제거 후 실제 QWEN VL이 생성하는 다양한 프롬프트 확인
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.qwen_wrapper import QwenWrapper
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_qwen_variety():
    """10가지 다양한 user prompt로 QWEN VL 다양성 테스트"""
    
    print("🧪 QWEN VL Prompt Enhancement Variety Test")
    print("=" * 60)
    
    # QWEN 모델 초기화
    print("📚 Initializing QWEN VL model...")
    qwen = QwenWrapper()
    
    # 테스트할 10가지 다양한 user prompt
    test_prompts = [
        "cat sitting",
        "beautiful sunset",
        "futuristic car",
        "ancient temple",
        "astronaut",
        "dragon",
        "cyberpunk city",
        "mountain landscape",
        "magical forest",
        "robot dancing"
    ]
    
    print(f"\n🎯 Testing {len(test_prompts)} different user prompts...\n")
    
    results = []
    
    for i, user_prompt in enumerate(test_prompts, 1):
        print(f"[{i:2d}/10] Testing: '{user_prompt}'")
        print("-" * 40)
        
        try:
            # QWEN VL로 프롬프트 향상
            result = qwen.enhance_prompt(user_prompt)
            
            original = result['original_prompt']
            enhanced = result['enhanced_prompt']
            raw_output = result['raw_output']
            
            # 결과 저장
            results.append({
                'original': original,
                'enhanced': enhanced,
                'raw_output': raw_output,
                'enhancement_ratio': len(enhanced) / len(original)
            })
            
            print(f"✅ Original:  '{original}'")
            print(f"✨ Enhanced:  '{enhanced}'")
            print(f"📊 Length ratio: {len(enhanced) / len(original):.1f}x")
            print(f"🔍 Raw output: '{raw_output[:100]}{'...' if len(raw_output) > 100 else ''}'")
            print()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'original': user_prompt,
                'enhanced': f"ERROR: {e}",
                'raw_output': str(e),
                'enhancement_ratio': 1.0
            })
            print()
    
    # 결과 요약
    print("=" * 60)
    print("📊 ENHANCEMENT SUMMARY")
    print("=" * 60)
    
    successful_results = [r for r in results if not r['enhanced'].startswith('ERROR:')]
    
    if successful_results:
        avg_ratio = sum(r['enhancement_ratio'] for r in successful_results) / len(successful_results)
        print(f"✅ Success rate: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)")
        print(f"📈 Average enhancement ratio: {avg_ratio:.1f}x")
        print(f"🎯 Min enhancement: {min(r['enhancement_ratio'] for r in successful_results):.1f}x")
        print(f"🚀 Max enhancement: {max(r['enhancement_ratio'] for r in successful_results):.1f}x")
        
        print("\n🔍 DIVERSITY ANALYSIS:")
        
        # 향상된 프롬프트의 다양성 분석
        enhanced_prompts = [r['enhanced'] for r in successful_results]
        
        # 공통 키워드 분석
        common_words = {}
        for prompt in enhanced_prompts:
            words = prompt.lower().split()
            for word in words:
                if len(word) > 3:  # 3글자 이상만
                    common_words[word] = common_words.get(word, 0) + 1
        
        # 가장 자주 사용된 키워드 top 5
        top_words = sorted(common_words.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"🔤 Most common keywords: {', '.join([f'{word}({count})' for word, count in top_words])}")
        
        # 프롬프트 길이 분포
        lengths = [len(r['enhanced']) for r in successful_results]
        print(f"📏 Enhanced prompt lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")
        
    else:
        print("❌ No successful enhancements!")
    
    print("\n✅ Variety test completed!")
    return results

if __name__ == "__main__":
    test_qwen_variety() 