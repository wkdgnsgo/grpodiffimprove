#!/usr/bin/env python3
"""
QWEN 통합 GRPO VLM 학습 메인 스크립트 (Accelerate 멀티 GPU 버전)
QWEN 모델의 enhance_prompt 기능과 GRPO를 통합한 프롬프트 개선 시스템
"""

import sys
import os
import logging
import torch
from pathlib import Path
from accelerate import Accelerator

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

def load_stable_diffusion_pipeline(device="cuda:5"):
    """Stable Diffusion 3 파이프라인 로드 (GPU 5번 전용)"""
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
    """학습용 프롬프트 데이터셋 (메모리 최적화 - 개수 축소)"""
    import random
    
    # 메모리 절약을 위해 프롬프트 수 축소
    selected_prompts = [
        # 기본적인 프롬프트들 (메모리 절약)
        "a beautiful cat sitting on a chair",
        "sunset over mountains with golden light",
        "abstract art painting with vibrant colors",
        "portrait of a woman with flowing hair",
        "futuristic city skyline at night",
        
        # 도전적인 프롬프트들
        "red apple on blue table with green background",
        "transparent glass sphere floating in purple space",
        "crowded marketplace with many people and colorful stalls"
    ]
    
    # 매번 다른 순서로 섞어서 반환
    random.shuffle(selected_prompts)
    
    return selected_prompts

def main():
    """메인 학습 함수 (Accelerate 멀티 GPU 버전)"""
    logger.info("🚀 QWEN 통합 GRPO VLM 학습 시작 (Accelerate 멀티 GPU)")
    logger.info("=" * 80)
    
    # Accelerate 초기화
    accelerator = Accelerator()
    logger.info(f"🎯 Accelerate 초기화 완료")
    logger.info(f"  - 프로세스 수: {accelerator.num_processes}")
    logger.info(f"  - 로컬 프로세스 인덱스: {accelerator.local_process_index}")
    logger.info(f"  - 디바이스: {accelerator.device}")
    
    # GPU 확인 및 배치 계획
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA 사용 가능 - GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"    메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB")
        
        logger.info("\n🎯 GPU 배치 계획 (Accelerate 2-GPU - GPU 0 부담 감소):")
        logger.info("  GPU 0-1: QWEN RL 학습 (Accelerate 분산 학습)")
        logger.info("  GPU 2: CLIP + QWEN Reference (통합)")
        logger.info("  GPU 3: Stable Diffusion 3 (이미지 생성 전용)")
    else:
        logger.warning("⚠️ CUDA 사용 불가 - CPU로 실행")
    
    # QWEN GRPO 설정 (Accelerate 멀티 GPU) - GPU 0 OOM 방지
    config = QWENGRPOConfig(
        learning_rate=1e-6,
        batch_size=1,  # GPU 0 OOM 방지를 위해 배치 크기 1로 축소
        num_rollouts=1,  # GPU 0 메모리 절약을 위해 롤아웃 수 1로 축소
        max_prompt_length=77,
        max_new_tokens=20,  # 토큰 수 더 축소
        temperature=1.2,
        top_p=0.9,
        top_k=100,
        kl_coef=0.02,
        clip_ratio=0.2,
        entropy_coef=0.01,
        save_images=True,
        log_dir="qwen_grpo_results"
    )
    
    logger.info("📋 QWEN GRPO 설정 (Accelerate 멀티 GPU):")
    logger.info(f"  - 학습률: {config.learning_rate}")
    logger.info(f"  - 배치 크기: {config.batch_size} (멀티 GPU)")
    logger.info(f"  - 롤아웃 수: {config.num_rollouts} (멀티 GPU)")
    logger.info(f"  - 온도: {config.temperature}")
    logger.info(f"  - KL 계수: {config.kl_coef}")
    
    try:
        # GPU 메모리 최적화 설정 (GPU 0 OOM 방지)
        if torch.cuda.is_available():
            # 메모리 할당 최적화
            torch.cuda.empty_cache()
            
            # GPU 0 메모리 제한 설정 (GPU 0 OOM 방지)
            if accelerator.local_process_index == 0:
                # GPU 0은 메인 프로세스로 메모리 사용량이 높으므로 제한
                torch.cuda.set_per_process_memory_fraction(0.8, device=0)
                logger.info("🔧 GPU 0 메모리 제한 설정: 80% (메인 프로세스)")
            else:
                # 다른 GPU들은 90% 사용 가능
                device_idx = accelerator.local_process_index
                torch.cuda.set_per_process_memory_fraction(0.9, device=device_idx)
                logger.info(f"🔧 GPU {device_idx} 메모리 제한 설정: 90%")
            
            # PyTorch 메모리 할당 최적화
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            logger.info("🧹 초기 GPU 메모리 정리 및 최적화 완료")
        
        # 1. QWEN VL 모델 로드 (Accelerate로 분산)
        logger.info("\n🧠 QWEN VL 모델 + GRPO 로딩... (Accelerate 분산)")
        qwen_model = QWENModel(
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            device="accelerate",  # Accelerate 전용 모드
            temperature=0.7,
            grpo_config=config,  # GRPO 컴포넌트 활성화
            is_main_process=accelerator.is_main_process  # Reference 모델 생성 여부
        )
        
        # Accelerate로 모델 준비 (단일 호출로 변경)
        logger.info("🔧 Accelerate를 통한 모델 분산 설정...")
        qwen_model.model = accelerator.prepare(qwen_model.model)
        qwen_model.grpo_optimizer = accelerator.prepare(qwen_model.grpo_optimizer)
        
        logger.info("✅ QWEN VL + GRPO 모델 로드 완료 (Accelerate 분산)")
        
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 QWEN 로드 후 메모리 정리")
        
        # 2. 통합 모델들 로드 (GPU 4번, 5번) - 메인 프로세스에서만
        logger.info("\n🎯 통합 모델들 로딩 체크...")
        
        # 메인 프로세스에서만 로딩
        if accelerator.is_main_process:
            logger.info("🎯 메인 프로세스: 통합 모델들 로딩 (GPU 2번, 3번)")
            
            # CLIP 리워드 모델 (GPU 2번)
            reward_model = CLIPReward(device="cuda:2")
            logger.info("✅ CLIP 리워드 모델 로드 완료 (GPU 2)")
            
            # Stable Diffusion 3 파이프라인 (GPU 3번) - 1개만 로딩
            sd_pipeline = load_stable_diffusion_pipeline(device="cuda:3")
            logger.info("✅ SD3 파이프라인 로드 완료 (GPU 3) - 1개만 로딩")
            
            # QWEN Reference 모델은 이미 qwen.py에서 GPU 2번으로 설정됨
            if hasattr(qwen_model, 'ref_model') and qwen_model.ref_model is not None:
                logger.info("✅ QWEN Reference 모델이 이미 GPU 2에 설정됨")
        else:
            # 서브 프로세스에서는 로딩하지 않음
            logger.info("🎯 서브 프로세스: 통합 모델들 로딩 건너뛰기")
            reward_model = None
            sd_pipeline = None
        
        # 모든 프로세스 동기화 (메인 프로세스의 모델 로딩 완료 대기)
        accelerator.wait_for_everyone()
        
        # 최종 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 모든 모델 로드 후 메모리 정리")
        
        # 3. QWEN GRPO 트레이너 초기화 (Accelerate 버전)
        logger.info("\n🎯 QWEN GRPO 트레이너 초기화... (Accelerate)")
        trainer = QWENGRPOTrainer(qwen_model, reward_model, sd_pipeline, config)
        
        # Accelerator를 트레이너에 전달
        trainer.accelerator = accelerator
        
        logger.info("✅ 트레이너 초기화 완료 (Accelerate)")
        
        # 4. 학습 데이터 준비
        train_prompts = get_training_prompts()
        logger.info(f"\n📝 학습 프롬프트: {len(train_prompts)}개")
        for i, prompt in enumerate(train_prompts):
            logger.info(f"  {i+1}. '{prompt}'")
        
        # 5. 베이스라인 성능 측정 (메인 프로세스에서만)
        if accelerator.is_main_process:
            logger.info("\n📊 베이스라인 성능 측정...")
            baseline_rewards = []
            
            # 첫 번째 프롬프트로 테스트
            test_prompt = train_prompts[0]
            logger.info(f"  베이스라인 테스트: '{test_prompt}'")
            
            try:
                # 기본 QWEN 향상
                with accelerator.device:
                    basic_result = qwen_model.enhance_prompt(test_prompt)
                    enhanced_prompt = basic_result['enhanced_prompt']
                
                # 이미지 생성 및 리워드 계산
                state = trainer.env.reset(test_prompt)
                trainer.env.current_enhanced_prompt = enhanced_prompt
                
                # 이미지 생성 (GPU 5번)
                with torch.cuda.device(5):
                    enhanced_result = sd_pipeline(
                        prompt=enhanced_prompt,
                        num_inference_steps=20,
                        guidance_scale=7.0,
                        height=1024,
                        width=1024
                    )
                    enhanced_image = enhanced_result.images[0]
                
                # 리워드 계산 (GPU 4번)
                with torch.cuda.device(4):
                    reward = reward_model.calculate_reward(
                        test_prompt,
                        enhanced_prompt,
                        enhanced_image
                    )
                
                baseline_rewards.append(reward)
                logger.info(f"    베이스라인 리워드: {reward:.3f}")
                
            except Exception as e:
                logger.warning(f"    베이스라인 측정 실패: {e}")
                baseline_rewards.append(0.5)  # 기본값
            
            avg_baseline = sum(baseline_rewards) / len(baseline_rewards) if baseline_rewards else 0.5
            logger.info(f"📈 베이스라인 평균 리워드: {avg_baseline:.3f}")
        else:
            avg_baseline = 0.5  # 다른 프로세스는 기본값
        
        # 베이스라인 값을 모든 프로세스에 브로드캐스트
        if accelerator.num_processes > 1:
            avg_baseline_tensor = torch.tensor(avg_baseline, device=accelerator.device)
            avg_baseline = accelerator.gather(avg_baseline_tensor)[0].item()
        
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 베이스라인 측정 후 메모리 정리")
        
        # 6. QWEN GRPO 학습 실행 (Accelerate 분산)
        logger.info("\n🚀 QWEN GRPO 학습 시작... (Accelerate 분산)")
        logger.info("=" * 80)
        
        all_metrics, baseline_data = trainer.train(
            train_prompts=train_prompts, 
            num_epochs=8,  # 멀티 GPU로 에포크 수 증가
            num_baseline_episodes=2
        )
        
        logger.info("✅ 학습 완료!")
        
        # 7. 학습 후 성능 측정 (메인 프로세스에서만)
        if accelerator.is_main_process:
            logger.info("\n📊 학습 후 성능 측정...")
            trained_rewards = []
            
            try:
                # GRPO 기반 향상
                with accelerator.device:
                    grpo_enhanced, log_prob = qwen_model.generate_grpo_enhanced_prompt(test_prompt)
                
                # 이미지 생성 및 리워드 계산
                state = trainer.env.reset(test_prompt)
                trainer.env.current_enhanced_prompt = grpo_enhanced
                
                # 이미지 생성 (GPU 5번)
                with torch.cuda.device(5):
                    enhanced_result = sd_pipeline(
                        prompt=grpo_enhanced,
                        num_inference_steps=20,
                        guidance_scale=7.0,
                        height=1024,
                        width=1024
                    )
                    enhanced_image = enhanced_result.images[0]
                
                # 리워드 계산 (GPU 4번)
                with torch.cuda.device(4):
                    reward = reward_model.calculate_reward(
                        test_prompt,
                        grpo_enhanced,
                        enhanced_image
                    )
                
                trained_rewards.append(reward)
                logger.info(f"    학습 후 리워드: {reward:.3f}")
                
            except Exception as e:
                logger.warning(f"    학습 후 평가 실패: {e}")
                trained_rewards.append(avg_baseline)  # 기본값
            
            avg_trained = sum(trained_rewards) / len(trained_rewards) if trained_rewards else avg_baseline
            logger.info(f"📈 학습 후 평균 리워드: {avg_trained:.3f}")
            
            # 8. 결과 분석 및 저장 (메인 프로세스에서만)
            logger.info("\n📋 최종 결과:")
            logger.info("=" * 80)
            logger.info(f"🎯 QWEN GRPO 학습 결과 (Accelerate 멀티 GPU)")
            logger.info(f"📊 베이스라인 리워드: {avg_baseline:.3f}")
            logger.info(f"📈 학습 후 리워드: {avg_trained:.3f}")
            logger.info(f"🔄 개선도: {avg_trained - avg_baseline:.3f}")
            
            if avg_baseline > 0:
                logger.info(f"📈 개선률: {((avg_trained - avg_baseline) / avg_baseline * 100):.1f}%")
            
            if avg_trained > avg_baseline:
                logger.info("✅ GRPO 학습이 성공적으로 개선되었습니다!")
            else:
                logger.info("⚠️ GRPO 학습 개선이 미미합니다. 더 많은 학습이 필요할 수 있습니다.")
            
            # 9. 모델 저장 (메인 프로세스에서만)
            logger.info("\n💾 모델 저장...")
            save_dir = Path("checkpoints")
            save_dir.mkdir(exist_ok=True)
            
            # Accelerate unwrap으로 원본 모델 저장
            unwrapped_model = accelerator.unwrap_model(qwen_model.model)
            
            model_path = save_dir / "qwen_grpo_model.pth"
            torch.save({
                'model_state_dict': unwrapped_model.state_dict(),
                'config': config,
                'baseline_reward': avg_baseline,
                'trained_reward': avg_trained,
                'improvement': avg_trained - avg_baseline,
                'training_metrics': all_metrics
            }, model_path)
            
            logger.info(f"✅ 모델 저장 완료: {model_path}")
        
        # 최종 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 최종 GPU 메모리 정리 완료")
        
        logger.info("\n🎉 QWEN GRPO 학습 완료!")
        
    except Exception as e:
        logger.error(f"❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
        # 에러 발생 시에도 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 에러 발생 후 GPU 메모리 정리")
        
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("\n🎉 프로그램이 성공적으로 완료되었습니다!")
    else:
        logger.error("\n❌ 프로그램 실행 중 오류가 발생했습니다.")
        sys.exit(1)

            