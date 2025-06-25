#!/usr/bin/env python3
"""
순수 GRPO VLM 학습 메인 스크립트
GPU 환경에서 실제 QWEN VL, Stable Diffusion 3, CLIP 모델을 사용한 학습
"""

import os
import sys
import logging
import torch
from pathlib import Path

# 현재 디렉토리를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from trainer_grpo_pure import PureGRPOConfig, PureGRPOTrainer
from qwen import QWENModel
from clip_reward import CLIPReward

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('grpo_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_stable_diffusion_pipeline(device="cuda:1"):
    """Stable Diffusion 3 파이프라인 로드 (GPU 1번)"""
    try:
        from diffusers import StableDiffusion3Pipeline
        import torch
        
        logger.info(f"🎨 Stable Diffusion 3 파이프라인 로딩... (Device: {device})")
        
        # GPU 메모리가 충분하지 않을 경우를 대비한 설정
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        
        # 지정된 GPU로 이동
        if torch.cuda.is_available():
            pipe = pipe.to(device)
            logger.info(f"✅ SD3 파이프라인을 {device}로 이동")
        else:
            logger.warning("⚠️ CUDA 사용 불가, CPU 사용")
        
        # 메모리 최적화
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        
        # 프로그레스 바 비활성화
        pipe.set_progress_bar_config(disable=True)
        
        logger.info("✅ Stable Diffusion 3 파이프라인 로드 완료")
        return pipe
        
    except Exception as e:
        logger.error(f"❌ SD3 파이프라인 로드 실패: {e}")
        raise

def get_training_prompts():
    """학습용 프롬프트 데이터셋"""
    return [
        # 기본 프롬프트
        "a beautiful cat sitting on a chair",
        "sunset over mountains with golden light",
        "abstract art painting with vibrant colors",
        "portrait of a woman with flowing hair",
        "futuristic city skyline at night",
        
        # 도전적인 프롬프트 (SD3가 어려워하는 것들)
        "red apple on blue table with green background",
        "transparent glass sphere floating in purple space",
        "wooden texture mixed with metallic surface",
        "fire and ice elements combined in one scene",
        "microscopic view of crystal structure",
        
        # 복잡한 장면
        "crowded marketplace with many people and colorful stalls",
        "underwater scene with coral reef and tropical fish",
        "ancient temple ruins covered with jungle vegetation",
        "steampunk mechanical device with gears and pipes",
        "surreal landscape with floating islands and waterfalls"
    ]

def main():
    """메인 학습 함수"""
    logger.info("🚀 순수 GRPO VLM 학습 시작")
    logger.info("=" * 80)
    
    # GPU 확인 및 배치 계획
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA 사용 가능 - GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        logger.info("\n🎯 GPU 배치 계획:")
        logger.info("  GPU 0: QWEN VL 모델 (프롬프트 향상)")
        logger.info("  GPU 1: Stable Diffusion 3 (이미지 생성)")
        logger.info("  GPU 2: CLIP 리워드 모델 (리워드 계산)")
    else:
        logger.warning("⚠️ CUDA 사용 불가 - CPU로 실행")
    
    # 설정
    config = PureGRPOConfig(
        learning_rate=1e-6,
        batch_size=4,
        num_rollouts=5,
        max_prompt_length=77,
        max_new_tokens=20,
        temperature=1.2,
        top_p=0.9,
        top_k=100,
        kl_coef=0.02,
        clip_ratio=0.2,
        entropy_coef=0.01
    )
    
    logger.info("📋 학습 설정:")
    logger.info(f"  - 학습률: {config.learning_rate}")
    logger.info(f"  - 배치 크기: {config.batch_size}")
    logger.info(f"  - 롤아웃 수: {config.num_rollouts}")
    logger.info(f"  - 최대 토큰: {config.max_new_tokens}")
    logger.info(f"  - 온도: {config.temperature}")
    logger.info(f"  - KL 계수: {config.kl_coef}")
    
    try:
        # 1. QWEN VL 모델 로드 (GPU 0번)
        logger.info("\n🧠 QWEN VL 모델 로딩...")
        qwen_model = QWENModel(device="cuda:0")
        logger.info("✅ QWEN VL 모델 로드 완료 (GPU 0)")
        
        # 2. CLIP 리워드 모델 로드 (GPU 2번)
        logger.info("\n🎯 CLIP 리워드 모델 로딩...")
        reward_model = CLIPReward(device="cuda:2")
        logger.info("✅ CLIP 리워드 모델 로드 완료 (GPU 2)")
        
        # 3. Stable Diffusion 3 파이프라인 로드 (GPU 1번)
        logger.info("\n🎨 Stable Diffusion 3 파이프라인 로딩...")
        sd_pipeline = load_stable_diffusion_pipeline(device="cuda:1")
        logger.info("✅ SD3 파이프라인 로드 완료 (GPU 1)")
        
        # 4. 순수 GRPO 트레이너 초기화
        logger.info("\n🎯 순수 GRPO 트레이너 초기화...")
        trainer = PureGRPOTrainer(qwen_model, reward_model, sd_pipeline, config)
        logger.info("✅ 트레이너 초기화 완료")
        
        # 5. 학습 데이터 준비
        train_prompts = get_training_prompts()
        logger.info(f"\n📝 학습 프롬프트: {len(train_prompts)}개")
        for i, prompt in enumerate(train_prompts[:5]):  # 처음 5개만 표시
            logger.info(f"  {i+1}. '{prompt}'")
        if len(train_prompts) > 5:
            logger.info(f"  ... 총 {len(train_prompts)}개")
        
        # 6. 베이스라인 성능 측정
        logger.info("\n📊 베이스라인 성능 측정...")
        baseline_rewards = []
        
        for i, prompt in enumerate(train_prompts[:3]):  # 처음 3개로 베이스라인 측정
            logger.info(f"  테스트 {i+1}/3: '{prompt}'")
            
            state = trainer.env.reset(prompt)
            original_prompt = trainer.env.current_prompt
            
            # 몇 스텝 실행
            for _ in range(config.max_new_tokens):
                action, _, _ = trainer.policy.get_action_and_log_prob(state)
                state, reward, done, info = trainer.env.step(action)
                if done:
                    baseline_rewards.append(reward)
                    enhanced_prompt = info['current_prompt']
                    logger.info(f"    '{original_prompt}' -> '{enhanced_prompt}' (reward: {reward:.3f})")
                    break
        
        avg_baseline = sum(baseline_rewards) / len(baseline_rewards) if baseline_rewards else 0.0
        logger.info(f"📈 베이스라인 평균 리워드: {avg_baseline:.3f}")
        
        # 7. GRPO 학습 실행
        logger.info("\n🚀 순수 GRPO 학습 시작...")
        logger.info("=" * 80)
        
        num_epochs = 10
        trainer.train(train_prompts, num_epochs=num_epochs)
        
        logger.info("✅ 학습 완료!")
        
        # 8. 학습 후 성능 측정
        logger.info("\n📊 학습 후 성능 측정...")
        trained_rewards = []
        
        for i, prompt in enumerate(train_prompts[:3]):  # 같은 프롬프트로 평가
            logger.info(f"  평가 {i+1}/3: '{prompt}'")
            
            state = trainer.env.reset(prompt)
            original_prompt = trainer.env.current_prompt
            
            # 몇 스텝 실행
            for _ in range(config.max_new_tokens):
                action, _, _ = trainer.policy.get_action_and_log_prob(state)
                state, reward, done, info = trainer.env.step(action)
                if done:
                    trained_rewards.append(reward)
                    enhanced_prompt = info['current_prompt']
                    logger.info(f"    '{original_prompt}' -> '{enhanced_prompt}' (reward: {reward:.3f})")
                    break
        
        avg_trained = sum(trained_rewards) / len(trained_rewards) if trained_rewards else 0.0
        logger.info(f"📈 학습 후 평균 리워드: {avg_trained:.3f}")
        
        # 9. 결과 분석 및 저장
        logger.info("\n📋 최종 결과:")
        logger.info("=" * 80)
        logger.info(f"🎯 순수 GRPO 학습 결과 (Value Network 없음)")
        logger.info(f"📊 베이스라인 리워드: {avg_baseline:.3f}")
        logger.info(f"📈 학습 후 리워드: {avg_trained:.3f}")
        logger.info(f"🔄 개선도: {avg_trained - avg_baseline:.3f}")
        logger.info(f"📈 개선률: {((avg_trained - avg_baseline) / avg_baseline * 100):.1f}%")
        
        if avg_trained > avg_baseline:
            logger.info("✅ 학습이 성공적으로 개선되었습니다!")
        else:
            logger.info("⚠️ 학습 개선이 미미합니다. 하이퍼파라미터 조정이 필요할 수 있습니다.")
        
        # 10. 모델 저장
        logger.info("\n💾 모델 저장...")
        save_dir = Path("checkpoints")
        save_dir.mkdir(exist_ok=True)
        
        model_path = save_dir / "pure_grpo_policy.pth"
        torch.save({
            'policy_state_dict': trainer.policy.state_dict(),
            'config': config,
            'baseline_reward': avg_baseline,
            'trained_reward': avg_trained,
            'improvement': avg_trained - avg_baseline
        }, model_path)
        
        logger.info(f"✅ 모델 저장 완료: {model_path}")
        
        logger.info("\n🎉 순수 GRPO 학습 완료!")
        
    except Exception as e:
        logger.error(f"❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("\n🎉 프로그램이 성공적으로 완료되었습니다!")
    else:
        logger.error("\n❌ 프로그램 실행 중 오류가 발생했습니다.")
        sys.exit(1)

            