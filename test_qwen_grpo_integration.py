#!/usr/bin/env python3
"""
QWEN GRPO 통합 시스템 테스트 스크립트
기본 기능들이 제대로 작동하는지 확인
"""

import torch
import logging
from qwen import QWENModel, QWENGRPOConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_qwen_grpo_integration():
    """QWEN GRPO 통합 시스템 테스트"""
    logger.info("🧪 QWEN GRPO 통합 시스템 테스트 시작")
    
    # 테스트용 설정
    config = QWENGRPOConfig(
        learning_rate=1e-6,
        batch_size=2,
        num_rollouts=2,
        num_enhancement_candidates=3,  # 3개 후보로 테스트
        save_images=False  # 이미지 저장 비활성화
    )
    
    try:
        # 1. QWEN 모델 로드 (GRPO 통합)
        logger.info("🧠 QWEN + GRPO 모델 로딩...")
        qwen_model = QWENModel(
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            temperature=0.7,
            grpo_config=config
        )
        logger.info("✅ 모델 로드 완료")
        
        # 2. 기본 프롬프트 향상 테스트
        test_prompt = "a beautiful sunset over mountains"
        logger.info(f"\n📝 테스트 프롬프트: '{test_prompt}'")
        
        # 기본 enhance_prompt 테스트
        logger.info("🔍 기본 enhance_prompt 테스트...")
        basic_result = qwen_model.enhance_prompt(test_prompt)
        logger.info(f"  원본: {test_prompt}")
        logger.info(f"  향상: {basic_result['enhanced_prompt']}")
        
        # 3. 후보 생성 테스트
        logger.info("\n🎯 후보 생성 테스트...")
        candidates = qwen_model.generate_enhancement_candidates(test_prompt)
        logger.info(f"  생성된 후보 수: {len(candidates)}")
        for i, candidate in enumerate(candidates):
            logger.info(f"  후보 {i}: {candidate}")
        
        # 4. GRPO 액션 선택 테스트
        logger.info("\n🎲 GRPO 액션 선택 테스트...")
        action, log_prob, action_candidates = qwen_model.get_grpo_action_and_log_prob(test_prompt)
        logger.info(f"  선택된 액션: {action}")
        logger.info(f"  로그 확률: {log_prob:.4f}")
        logger.info(f"  선택된 프롬프트: {action_candidates[action]}")
        
        # 5. 상태 표현 테스트
        logger.info("\n🧮 상태 표현 테스트...")
        state_repr = qwen_model.get_grpo_state_representation(test_prompt)
        logger.info(f"  상태 표현 크기: {state_repr.shape}")
        logger.info(f"  상태 표현 타입: {state_repr.dtype}")
        
        # 6. 참조 정책 테스트
        logger.info("\n📊 참조 정책 테스트...")
        ref_log_prob = qwen_model.get_ref_policy_log_prob(test_prompt, action)
        logger.info(f"  참조 정책 로그 확률: {ref_log_prob:.4f}")
        
        # 7. 간단한 경험 데이터로 업데이트 테스트
        logger.info("\n🎯 GRPO 업데이트 테스트...")
        fake_experiences = [
            {
                'user_prompt': test_prompt,
                'action': action,
                'log_prob': log_prob,
                'reward': 0.5
            },
            {
                'user_prompt': test_prompt,
                'action': (action + 1) % len(action_candidates),
                'log_prob': log_prob * 0.9,
                'reward': 0.3
            }
        ]
        
        metrics = qwen_model.update_grpo_policy(fake_experiences)
        logger.info("  업데이트 메트릭:")
        for key, value in metrics.items():
            logger.info(f"    {key}: {value:.4f}")
        
        logger.info("\n✅ 모든 테스트 통과!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prompt_quality():
    """프롬프트 품질 비교 테스트"""
    logger.info("\n🔍 프롬프트 품질 비교 테스트")
    
    config = QWENGRPOConfig(num_enhancement_candidates=5, save_images=False)
    
    try:
        qwen_model = QWENModel(
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            grpo_config=config
        )
        
        test_prompts = [
            "a cat",
            "beautiful landscape",
            "futuristic city",
            "abstract art"
        ]
        
        for prompt in test_prompts:
            logger.info(f"\n📝 테스트: '{prompt}'")
            
            # 기본 향상
            basic = qwen_model.enhance_prompt(prompt)
            logger.info(f"  기본: {basic['enhanced_prompt']}")
            
            # GRPO 후보들
            candidates = qwen_model.generate_enhancement_candidates(prompt)
            logger.info("  GRPO 후보들:")
            for i, candidate in enumerate(candidates):
                logger.info(f"    {i}: {candidate}")
            
            # GRPO 선택
            action, _, _ = qwen_model.get_grpo_action_and_log_prob(prompt)
            logger.info(f"  GRPO 선택 (액션 {action}): {candidates[action]}")
        
        logger.info("\n✅ 품질 비교 테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 품질 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 QWEN GRPO 통합 테스트 시작")
    logger.info("=" * 60)
    
    # 기본 기능 테스트
    success1 = test_qwen_grpo_integration()
    
    # 품질 비교 테스트
    success2 = test_prompt_quality()
    
    if success1 and success2:
        logger.info("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
    else:
        logger.error("\n❌ 일부 테스트가 실패했습니다.") 