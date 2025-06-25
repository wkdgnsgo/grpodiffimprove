#!/usr/bin/env python3
"""
QWEN 통합 GRPO VLM 학습 메인 스크립트
QWEN 모델의 enhance_prompt 기능과 GRPO를 통합한 프롬프트 개선 시스템
"""

import sys
import logging
import torch
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('qwen_grpo_training.log')
    ]
)

logger = logging.getLogger(__name__)

# 모델 임포트
from qwen import QWENModel, QWENGRPOConfig
from clip_reward import CLIPReward
from trainer_grpo_pure import QWENGRPOTrainer

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
    """학습용 프롬프트 데이터셋 (다양성 확보)"""
    import random
    
    # 전체 프롬프트 풀
    all_prompts = [
        # 동물들
        "a beautiful cat sitting on a chair",
        "majestic lion in African savanna",
        "colorful parrot in tropical rainforest",
        "graceful swan on peaceful lake",
        "playful dolphin jumping in ocean",
        
        # 자연 풍경
        "sunset over mountains with golden light",
        "misty forest with tall pine trees",
        "desert landscape with sand dunes",
        "rocky coastline with crashing waves",
        "cherry blossoms in spring garden",
        
        # 예술과 추상
        "abstract art painting with vibrant colors",
        "geometric patterns in bright neon colors",
        "watercolor painting of flowers",
        "minimalist sculpture in white marble",
        "street art mural on brick wall",
        
        # 인물
        "portrait of a woman with flowing hair",
        "elderly man reading book by fireplace",
        "child playing in summer meadow",
        "dancer in elegant pose",
        "musician playing violin on stage",
        
        # 도시와 건축
        "futuristic city skyline at night",
        "ancient castle on mountain peak",
        "modern glass building reflecting sky",
        "cozy cafe with warm lighting",
        "busy train station with commuters",
        
        # 도전적인 프롬프트
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
    
    # 매번 다른 순서로 섞어서 반환 (다양성 확보)
    random.shuffle(all_prompts)
    
    # 처음 15개 선택 (충분한 다양성 + 적당한 크기)
    selected_prompts = all_prompts[:15]
    
    return selected_prompts

def main():
    """메인 학습 함수"""
    logger.info("🚀 QWEN 통합 GRPO VLM 학습 시작")
    logger.info("=" * 80)
    
    # GPU 확인 및 배치 계획
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA 사용 가능 - GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        logger.info("\n🎯 GPU 배치 계획:")
        logger.info("  GPU 0: QWEN VL 모델 + GRPO 정책 (프롬프트 향상 및 학습)")
        logger.info("  GPU 1: Stable Diffusion 3 (이미지 생성)")
        logger.info("  GPU 2: CLIP 리워드 모델 (리워드 계산)")
    else:
        logger.warning("⚠️ CUDA 사용 불가 - CPU로 실행")
    
    # QWEN GRPO 설정
            config = QWENGRPOConfig(
        learning_rate=1e-6,
        batch_size=4,
        num_rollouts=3,  # 롤아웃 수 줄임 (각 프롬프트당 3개 롤아웃)
        max_prompt_length=77,
        max_new_tokens=30,
        temperature=1.2,
        top_p=0.9,
        top_k=100,
        kl_coef=0.02,
        clip_ratio=0.2,
        entropy_coef=0.01,
        save_images=True,
        log_dir="qwen_grpo_results"
    )
    
    logger.info("📋 QWEN GRPO 설정:")
    logger.info(f"  - 학습률: {config.learning_rate}")
    logger.info(f"  - 배치 크기: {config.batch_size}")
    logger.info(f"  - 롤아웃 수: {config.num_rollouts}")
    logger.info(f"  - 롤아웃 수: {config.num_rollouts}")
    logger.info(f"  - 온도: {config.temperature}")
    logger.info(f"  - KL 계수: {config.kl_coef}")
    
    try:
        # 1. QWEN VL 모델 로드 (GRPO 통합) (GPU 0번)
        logger.info("\n🧠 QWEN VL 모델 + GRPO 로딩...")
        qwen_model = QWENModel(
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            device="cuda:0",
            temperature=0.7,
            grpo_config=config  # GRPO 컴포넌트 활성화
        )
        logger.info("✅ QWEN VL + GRPO 모델 로드 완료 (GPU 0)")
        
        # 2. CLIP 리워드 모델 로드 (GPU 2번)
        logger.info("\n🎯 CLIP 리워드 모델 로딩...")
        reward_model = CLIPReward(device="cuda:2")
        logger.info("✅ CLIP 리워드 모델 로드 완료 (GPU 2)")
        
        # 3. Stable Diffusion 3 파이프라인 로드 (GPU 1번)
        logger.info("\n🎨 Stable Diffusion 3 파이프라인 로딩...")
        sd_pipeline = load_stable_diffusion_pipeline(device="cuda:1")
        logger.info("✅ SD3 파이프라인 로드 완료 (GPU 1)")
        
        # 4. QWEN GRPO 트레이너 초기화
        logger.info("\n🎯 QWEN GRPO 트레이너 초기화...")
        trainer = QWENGRPOTrainer(qwen_model, reward_model, sd_pipeline, config)
        logger.info("✅ 트레이너 초기화 완료")
        
        # 5. 학습 데이터 준비
        train_prompts = get_training_prompts()
        logger.info(f"\n📝 학습 프롬프트: {len(train_prompts)}개")
        for i, prompt in enumerate(train_prompts[:5]):  # 처음 5개만 표시
            logger.info(f"  {i+1}. '{prompt}'")
        if len(train_prompts) > 5:
            logger.info(f"  ... 총 {len(train_prompts)}개")
        
        # 6. 베이스라인 성능 측정 (기본 QWEN enhance_prompt)
        logger.info("\n📊 베이스라인 성능 측정 (기본 QWEN)...")
        baseline_rewards = []
        
        # 다양한 프롬프트로 베이스라인 측정 (첫 번째, 중간, 마지막)
        baseline_test_indices = [0, len(train_prompts)//2, len(train_prompts)-1]
        baseline_test_prompts = [train_prompts[i] for i in baseline_test_indices]
        
        for i, prompt in enumerate(baseline_test_prompts):
            logger.info(f"  테스트 {i+1}/3: '{prompt}'")
            
            try:
                # 기본 QWEN 향상
                with torch.cuda.device(0):
                    basic_result = qwen_model.enhance_prompt(prompt)
                    enhanced_prompt = basic_result['enhanced_prompt']
                
                # 이미지 생성 및 리워드 계산
                state = trainer.env.reset(prompt)
                trainer.env.current_enhanced_prompt = enhanced_prompt
                
                # 이미지 생성
                with torch.cuda.device(1):
                    enhanced_result = sd_pipeline(
                        prompt=enhanced_prompt,
                        num_inference_steps=28,
                        guidance_scale=7.0,
                        height=1024,
                        width=1024
                    )
                    enhanced_image = enhanced_result.images[0]
                
                # 리워드 계산
                with torch.cuda.device(2):
                    reward = reward_model.calculate_reward(
                        prompt,
                        enhanced_prompt,
                        enhanced_image
                    )
                
                baseline_rewards.append(reward)
                logger.info(f"    '{prompt}' -> '{enhanced_prompt[:50]}...' (reward: {reward:.3f})")
                
            except Exception as e:
                logger.warning(f"    베이스라인 측정 실패: {e}")
                continue
        
        avg_baseline = sum(baseline_rewards) / len(baseline_rewards) if baseline_rewards else 0.0
        logger.info(f"📈 베이스라인 평균 리워드: {avg_baseline:.3f}")
        
        # 7. QWEN GRPO 학습 실행
        logger.info("\n🚀 QWEN GRPO 학습 시작...")
        logger.info("=" * 80)
        
        all_metrics, baseline_data = trainer.train(
            train_prompts=train_prompts, 
            num_epochs=10, 
            num_baseline_episodes=3  # 베이스라인 에피소드 수 조정 가능
        )
        
        logger.info("✅ 학습 완료!")
        
        # 8. 학습 후 성능 측정 (GRPO 기반)
        logger.info("\n📊 학습 후 성능 측정 (GRPO)...")
        trained_rewards = []
        
        # 베이스라인과 같은 프롬프트로 평가
        for i, prompt in enumerate(baseline_test_prompts):
            logger.info(f"  평가 {i+1}/3: '{prompt}'")
            
            try:
                # GRPO 기반 향상
                with torch.cuda.device(0):
                    grpo_enhanced, log_prob = qwen_model.generate_grpo_enhanced_prompt(prompt)
                
                # 이미지 생성 및 리워드 계산
                state = trainer.env.reset(prompt)
                trainer.env.current_enhanced_prompt = grpo_enhanced
                
                # 이미지 생성
                with torch.cuda.device(1):
                    enhanced_result = sd_pipeline(
                        prompt=grpo_enhanced,
                        num_inference_steps=28,
                        guidance_scale=7.0,
                        height=1024,
                        width=1024
                    )
                    enhanced_image = enhanced_result.images[0]
                
                # 리워드 계산
                with torch.cuda.device(2):
                    reward = reward_model.calculate_reward(
                        prompt,
                        grpo_enhanced,
                        enhanced_image
                    )
                
                trained_rewards.append(reward)
                logger.info(f"    '{prompt}' -> '{grpo_enhanced[:50]}...' (reward: {reward:.3f})")
                
            except Exception as e:
                logger.warning(f"    학습 후 평가 실패: {e}")
                continue
        
        avg_trained = sum(trained_rewards) / len(trained_rewards) if trained_rewards else 0.0
        logger.info(f"📈 학습 후 평균 리워드: {avg_trained:.3f}")
        
        # 9. 결과 분석 및 저장
        logger.info("\n📋 최종 결과:")
        logger.info("=" * 80)
        logger.info(f"🎯 QWEN GRPO 학습 결과")
        logger.info(f"📊 베이스라인 리워드 (기본 QWEN): {avg_baseline:.3f}")
        logger.info(f"📈 학습 후 리워드 (GRPO): {avg_trained:.3f}")
        logger.info(f"🔄 개선도: {avg_trained - avg_baseline:.3f}")
        
        if avg_baseline > 0:
            logger.info(f"📈 개선률: {((avg_trained - avg_baseline) / avg_baseline * 100):.1f}%")
        
        if avg_trained > avg_baseline:
            logger.info("✅ GRPO 학습이 성공적으로 개선되었습니다!")
        else:
            logger.info("⚠️ GRPO 학습 개선이 미미합니다. 하이퍼파라미터 조정이 필요할 수 있습니다.")
        
        # 10. 모델 저장
        logger.info("\n💾 모델 저장...")
        save_dir = Path("checkpoints")
        save_dir.mkdir(exist_ok=True)
        
        model_path = save_dir / "qwen_grpo_model.pth"
        torch.save({
            'model_state_dict': qwen_model.model.state_dict(),
            'config': config,
            'baseline_reward': avg_baseline,
            'trained_reward': avg_trained,
            'improvement': avg_trained - avg_baseline,
            'training_metrics': all_metrics
        }, model_path)
        
        logger.info(f"✅ 모델 저장 완료: {model_path}")
        
        logger.info("\n🎉 QWEN GRPO 학습 완료!")
        
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

            