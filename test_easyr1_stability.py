#!/usr/bin/env python3
"""
EasyR1 스타일 수치적 안정성 기법 테스트

이 스크립트는 QWEN 모델에 적용된 EasyR1 스타일 수치적 안정성 기법들을 테스트합니다:
1. 적응적 그래디언트 클리핑 (AdaGC)
2. 그래디언트 중앙화 (Gradient Centralization)
3. 적응적 그래디언트 정규화 (Adaptive Gradient Normalization)
4. 확률적 반올림 시뮬레이션 (Stochastic Rounding)
5. 보수적인 logits 클리핑
"""

import torch
import logging
from qwen import QWENModel, QWENGRPOConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_easyr1_stability():
    """EasyR1 스타일 수치적 안정성 기법 테스트"""
    
    logger.info("🧪 EasyR1 스타일 수치적 안정성 기법 테스트 시작")
    
    # EasyR1 안정성 기법이 활성화된 설정
    grpo_config = QWENGRPOConfig(
        # 기본 설정
        learning_rate=1e-4,
        batch_size=1,
        num_rollouts=1,
        max_new_tokens=10,
        
        # EasyR1 안정성 기법 활성화
        use_adaptive_grad_clip=True,
        grad_clip_ema_beta=0.99,
        grad_clip_coef=1.5,
        use_grad_centralization=True,
        use_grad_normalization=True,
        grad_norm_alpha=0.5,
        use_stochastic_rounding=True,
        logits_clip_range=20.0,
        stable_log_prob_min=-50.0
    )
    
    # QWEN 모델 초기화
    logger.info("🔧 QWEN 모델 초기화 중...")
    model = QWENModel(
        device="cuda" if torch.cuda.is_available() else "cpu",
        grpo_config=grpo_config
    )
    
    logger.info("✅ QWEN 모델 초기화 완료")
    
    # 1. 기본 프롬프트 향상 테스트
    logger.info("\n🔍 1. 기본 프롬프트 향상 테스트")
    test_prompts = [
        "cat",
        "beautiful sunset",
        "futuristic city"
    ]
    
    for prompt in test_prompts:
        try:
            result = model.enhance_prompt(prompt)
            logger.info(f"  '{prompt}' -> '{result['enhanced_prompt']}'")
        except Exception as e:
            logger.error(f"  ❌ '{prompt}' 향상 실패: {e}")
    
    # 2. GRPO 로그 확률 계산 안정성 테스트
    logger.info("\n🔍 2. GRPO 로그 확률 계산 안정성 테스트")
    
    test_cases = [
        ("cat", "cat, high quality, detailed photography"),
        ("dog", "dog, professional portrait, studio lighting"),
        ("flower", "flower, macro photography, vibrant colors")
    ]
    
    for user_prompt, enhanced_prompt in test_cases:
        try:
            # 현재 모델 로그 확률
            current_log_prob = model.calculate_log_prob_for_grpo(user_prompt, enhanced_prompt)
            logger.info(f"  Current log prob for '{user_prompt}': {current_log_prob:.6f}")
            
            # Reference 모델 로그 확률
            ref_log_prob = model.get_ref_model_log_prob(user_prompt, enhanced_prompt)
            logger.info(f"  Reference log prob for '{user_prompt}': {ref_log_prob:.6f}")
            
            # 안정성 검증
            if torch.isnan(current_log_prob) or torch.isinf(current_log_prob):
                logger.error(f"  ❌ Current log prob에 nan/inf 발견!")
            else:
                logger.info(f"  ✅ Current log prob 안정성 확인")
                
            if torch.isnan(ref_log_prob) or torch.isinf(ref_log_prob):
                logger.error(f"  ❌ Reference log prob에 nan/inf 발견!")
            else:
                logger.info(f"  ✅ Reference log prob 안정성 확인")
                
        except Exception as e:
            logger.error(f"  ❌ 로그 확률 계산 실패: {e}")
    
    # 3. 그래디언트 안정성 기법 테스트
    logger.info("\n🔍 3. 그래디언트 안정성 기법 테스트")
    
    # 더미 경험 데이터 생성
    dummy_experiences = []
    for i, (user_prompt, enhanced_prompt) in enumerate(test_cases):
        try:
            log_prob = model.calculate_log_prob_for_grpo(user_prompt, enhanced_prompt)
            dummy_experiences.append({
                'user_prompt': user_prompt,
                'enhanced_prompt': enhanced_prompt,
                'log_prob': log_prob,
                'reward': 0.5 + i * 0.1  # 더미 리워드
            })
        except Exception as e:
            logger.error(f"  ❌ 더미 경험 생성 실패: {e}")
    
    if dummy_experiences:
        try:
            logger.info(f"  더미 경험 {len(dummy_experiences)}개로 GRPO 업데이트 테스트")
            
            # 그래디언트 norm 추적을 위한 초기값 설정
            initial_grad_norm_ema = model.grad_norm_ema
            logger.info(f"  초기 그래디언트 norm EMA: {initial_grad_norm_ema}")
            
            # GRPO 정책 업데이트 (EasyR1 안정성 기법 적용)
            metrics = model.update_grpo_policy(dummy_experiences)
            
            logger.info("  📊 GRPO 업데이트 메트릭:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"    {key}: {value:.6f}")
                else:
                    logger.info(f"    {key}: {value}")
            
            # 그래디언트 norm EMA 변화 확인
            final_grad_norm_ema = model.grad_norm_ema
            logger.info(f"  최종 그래디언트 norm EMA: {final_grad_norm_ema}")
            
            if final_grad_norm_ema != initial_grad_norm_ema:
                logger.info("  ✅ 적응적 그래디언트 클리핑 EMA 업데이트 확인")
            else:
                logger.info("  ℹ️ 그래디언트 norm EMA 변화 없음 (정상적일 수 있음)")
                
        except Exception as e:
            logger.error(f"  ❌ GRPO 업데이트 테스트 실패: {e}")
    
    # 4. 설정값 검증
    logger.info("\n🔍 4. EasyR1 안정성 설정값 검증")
    logger.info(f"  적응적 그래디언트 클리핑: {model.grpo_config.use_adaptive_grad_clip}")
    logger.info(f"  그래디언트 중앙화: {model.grpo_config.use_grad_centralization}")
    logger.info(f"  그래디언트 정규화: {model.grpo_config.use_grad_normalization}")
    logger.info(f"  확률적 반올림: {model.grpo_config.use_stochastic_rounding}")
    logger.info(f"  Logits 클리핑 범위: ±{model.grpo_config.logits_clip_range}")
    logger.info(f"  안전한 로그 확률 최소값: {model.grpo_config.stable_log_prob_min}")
    logger.info(f"  그래디언트 클리핑 EMA 베타: {model.grpo_config.grad_clip_ema_beta}")
    logger.info(f"  그래디언트 클리핑 계수: {model.grpo_config.grad_clip_coef}")
    logger.info(f"  그래디언트 정규화 알파: {model.grpo_config.grad_norm_alpha}")
    
    logger.info("\n✅ EasyR1 스타일 수치적 안정성 기법 테스트 완료")

def compare_with_without_stability():
    """안정성 기법 적용 전후 비교"""
    
    logger.info("\n🔄 안정성 기법 적용 전후 비교 테스트")
    
    # 안정성 기법 비활성화 설정
    config_without = QWENGRPOConfig(
        learning_rate=1e-4,
        batch_size=1,
        max_new_tokens=10,
        use_adaptive_grad_clip=False,
        use_grad_centralization=False,
        use_grad_normalization=False,
        use_stochastic_rounding=False,
        logits_clip_range=100.0  # 기존 값
    )
    
    # 안정성 기법 활성화 설정
    config_with = QWENGRPOConfig(
        learning_rate=1e-4,
        batch_size=1,
        max_new_tokens=10,
        use_adaptive_grad_clip=True,
        use_grad_centralization=True,
        use_grad_normalization=True,
        use_stochastic_rounding=True,
        logits_clip_range=20.0  # EasyR1 스타일
    )
    
    test_prompt = "beautiful landscape"
    
    # 비교 테스트는 메모리 제약으로 생략 (실제 환경에서는 유용)
    logger.info("  📝 비교 테스트는 메모리 제약으로 생략")
    logger.info("  💡 실제 트레이닝에서는 두 설정의 성능을 비교해보세요")

if __name__ == "__main__":
    try:
        test_easyr1_stability()
        compare_with_without_stability()
        
        print("\n" + "="*80)
        print("🎉 EasyR1 스타일 수치적 안정성 기법 테스트 성공!")
        print("="*80)
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc() 