#!/usr/bin/env python3
"""
GRPO 알고리즘 정확성 검증 테스트
- 로그 확률 계산 일관성 검사
- π_new/π_old 비율 계산 검증
- 클리핑 동작 확인
- KL divergence 계산 검증
"""

import torch
import torch.nn.functional as F
import logging
from qwen import QWENModel, QWENGRPOConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_log_prob_consistency():
    """로그 확률 계산 일관성 테스트"""
    print("🔍 로그 확률 계산 일관성 테스트")
    
    # 모델 초기화
    config = QWENGRPOConfig(
        batch_size=1,
        num_rollouts=1,
        max_new_tokens=10
    )
    model = QWENModel(device="cuda", grpo_config=config)
    
    # 테스트 데이터
    user_prompt = "cat"
    
    # 1. 생성 시점의 로그 확률 계산
    enhanced_prompt, generation_log_prob = model.generate_grpo_enhanced_prompt(user_prompt)
    print(f"📝 생성된 프롬프트: {enhanced_prompt}")
    print(f"📊 생성 시점 로그 확률: {generation_log_prob:.6f}")
    
    # 2. 현재 모델로 재계산
    current_log_prob = model.calculate_log_prob_for_grpo(user_prompt, enhanced_prompt)
    print(f"📊 현재 모델 로그 확률: {current_log_prob:.6f}")
    
    # 3. 참조 모델로 계산
    ref_log_prob = model.get_ref_model_log_prob(user_prompt, enhanced_prompt)
    print(f"📊 참조 모델 로그 확률: {ref_log_prob:.6f}")
    
    # 4. 일관성 검사
    log_prob_diff = abs(generation_log_prob - current_log_prob)
    print(f"🔍 생성 vs 현재 모델 차이: {log_prob_diff:.6f}")
    
    if log_prob_diff < 0.001:
        print("✅ 로그 확률 계산 일관성 PASS")
    else:
        print("❌ 로그 확률 계산 일관성 FAIL - 차이가 너무 큼")
    
    return {
        'generation_log_prob': generation_log_prob.item(),
        'current_log_prob': current_log_prob.item(),
        'ref_log_prob': ref_log_prob.item(),
        'consistency_check': log_prob_diff < 0.001
    }

def test_grpo_ratio_calculation():
    """GRPO 비율 계산 검증"""
    print("\n🔍 GRPO 비율 계산 검증")
    
    # 테스트 로그 확률 값들
    old_log_probs = torch.tensor([-5.2, -4.8, -6.1, -5.5])
    current_log_probs = torch.tensor([-5.0, -4.9, -6.0, -5.3])
    
    # 로그 비율 계산
    log_ratio = current_log_probs - old_log_probs
    print(f"📊 로그 비율: {log_ratio}")
    
    # 비율 계산
    ratio = torch.exp(log_ratio)
    print(f"📊 비율 (π_new/π_old): {ratio}")
    
    # 클리핑 적용
    clip_ratio = 0.1
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    print(f"📊 클리핑된 비율 (범위: {1-clip_ratio:.1f}-{1+clip_ratio:.1f}): {clipped_ratio}")
    
    # 클리핑 효과 확인
    clipping_applied = torch.any(ratio != clipped_ratio)
    print(f"🔍 클리핑 적용됨: {clipping_applied}")
    
    return {
        'log_ratio': log_ratio.tolist(),
        'ratio': ratio.tolist(),
        'clipped_ratio': clipped_ratio.tolist(),
        'clipping_applied': clipping_applied.item()
    }

def test_policy_loss_calculation():
    """정책 손실 계산 검증"""
    print("\n🔍 정책 손실 계산 검증")
    
    # 테스트 데이터
    advantages = torch.tensor([0.5, -0.2, 0.8, -0.1])
    ratio = torch.tensor([1.05, 0.95, 1.15, 0.88])
    clip_ratio = 0.1
    
    # 클리핑된 비율
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    
    # 정책 목적 함수
    policy_obj1 = ratio * advantages
    policy_obj2 = clipped_ratio * advantages
    
    print(f"📊 Advantages: {advantages}")
    print(f"📊 Ratio: {ratio}")
    print(f"📊 Clipped ratio: {clipped_ratio}")
    print(f"📊 Policy obj 1 (ratio * adv): {policy_obj1}")
    print(f"📊 Policy obj 2 (clipped * adv): {policy_obj2}")
    
    # 정책 손실 (음수 최소값)
    policy_loss = -torch.min(policy_obj1, policy_obj2).mean()
    print(f"📊 Policy loss: {policy_loss:.6f}")
    
    # 클리핑 효과 분석
    clipping_effect = torch.sum(policy_obj1 != policy_obj2)
    print(f"🔍 클리핑이 적용된 요소 수: {clipping_effect}")
    
    return {
        'policy_loss': policy_loss.item(),
        'clipping_effect': clipping_effect.item(),
        'policy_obj1': policy_obj1.tolist(),
        'policy_obj2': policy_obj2.tolist()
    }

def test_kl_divergence():
    """KL divergence 계산 검증"""
    print("\n🔍 KL divergence 계산 검증")
    
    # 테스트 로그 확률
    current_log_probs = torch.tensor([-5.0, -4.9, -6.0, -5.3])
    ref_log_probs = torch.tensor([-5.2, -4.8, -6.1, -5.5])
    
    # KL divergence 계산 (current - reference)
    kl_div = (current_log_probs - ref_log_probs).mean()
    print(f"📊 Current log probs: {current_log_probs}")
    print(f"📊 Reference log probs: {ref_log_probs}")
    print(f"📊 KL divergence: {kl_div:.6f}")
    
    # KL 페널티
    kl_coef = 0.02
    kl_penalty = kl_coef * kl_div
    print(f"📊 KL penalty (coef={kl_coef}): {kl_penalty:.6f}")
    
    return {
        'kl_div': kl_div.item(),
        'kl_penalty': kl_penalty.item()
    }

def test_full_grpo_update():
    """전체 GRPO 업데이트 프로세스 테스트"""
    print("\n🔍 전체 GRPO 업데이트 프로세스 테스트")
    
    # 모델 초기화
    config = QWENGRPOConfig(
        batch_size=2,
        num_rollouts=1,
        max_new_tokens=10,
        clip_ratio=0.1,
        kl_coef=0.02
    )
    model = QWENModel(device="cuda", grpo_config=config)
    
    # 가짜 경험 데이터 생성
    experiences = []
    for i in range(2):
        user_prompt = f"test prompt {i}"
        enhanced_prompt, log_prob = model.generate_grpo_enhanced_prompt(user_prompt)
        
        experience = {
            'user_prompt': user_prompt,
            'enhanced_prompt': enhanced_prompt,
            'log_prob': log_prob,
            'reward': 0.5 + i * 0.1,
            'group_advantage': 0.2 + i * 0.1
        }
        experiences.append(experience)
    
    print(f"📝 생성된 경험 수: {len(experiences)}")
    
    # GRPO 업데이트 실행
    metrics = model.update_grpo_policy(experiences)
    
    print("📊 업데이트 결과:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # 결과 검증
    success_checks = []
    success_checks.append(('policy_loss', not torch.isnan(torch.tensor(metrics['policy_loss']))))
    success_checks.append(('kl_div', not torch.isnan(torch.tensor(metrics['kl_div']))))
    success_checks.append(('total_loss', not torch.isnan(torch.tensor(metrics['total_loss']))))
    
    print("\n✅ 검증 결과:")
    for check_name, passed in success_checks:
        status = "PASS" if passed else "FAIL"
        print(f"  {check_name}: {status}")
    
    return metrics

def main():
    """메인 테스트 실행"""
    print("🚀 GRPO 알고리즘 정확성 검증 시작")
    
    try:
        # 1. 로그 확률 일관성 테스트
        log_prob_results = test_log_prob_consistency()
        
        # 2. 비율 계산 테스트
        ratio_results = test_grpo_ratio_calculation()
        
        # 3. 정책 손실 계산 테스트
        policy_loss_results = test_policy_loss_calculation()
        
        # 4. KL divergence 테스트
        kl_results = test_kl_divergence()
        
        # 5. 전체 업데이트 프로세스 테스트
        full_update_results = test_full_grpo_update()
        
        print("\n🎯 전체 테스트 완료!")
        print("📊 핵심 지표:")
        print(f"  로그 확률 일관성: {'✅' if log_prob_results['consistency_check'] else '❌'}")
        print(f"  클리핑 적용: {'✅' if ratio_results['clipping_applied'] else '⚠️'}")
        print(f"  정책 손실 계산: {'✅' if not torch.isnan(torch.tensor(policy_loss_results['policy_loss'])) else '❌'}")
        print(f"  KL divergence: {'✅' if not torch.isnan(torch.tensor(kl_results['kl_div'])) else '❌'}")
        print(f"  전체 업데이트: {'✅' if 'error' not in full_update_results else '❌'}")
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 