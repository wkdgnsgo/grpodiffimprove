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

def load_stable_diffusion_pipeline(device="cuda:6"):
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
        
        logger.info("\n🎯 GPU 배치 계획 (DeepSpeed ZeRO Stage 3 전체 학습 - 8 GPU):")
        logger.info("  GPU 0-3: QWEN 전체 학습 (DeepSpeed ZeRO Stage 3 분산)")
        logger.info("  GPU 4-5: CLIP 리워드 모델 (이중화)")
        logger.info("  GPU 6-7: Stable Diffusion 3 (이중화)")
    else:
        logger.warning("⚠️ CUDA 사용 불가 - CPU로 실행")
    
    # QWEN GRPO 설정 (DeepSpeed ZeRO Stage 3 전체 학습)
    config = QWENGRPOConfig(
        learning_rate=5e-7,  # 전체 학습용 낮은 학습률
        batch_size=2,  # DeepSpeed ZeRO Stage 3로 배치 크기 증가
        num_rollouts=2,  # 4개 GPU로 롤아웃 수 증가
        max_prompt_length=77,
        max_new_tokens=25,  # 토큰 수 복원
        temperature=1.2,
        top_p=0.9,
        top_k=100,
        kl_coef=0.02,
        clip_ratio=0.2,
        entropy_coef=0.01,
        save_images=True,
        log_dir="qwen_grpo_full_training_results"
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
            
            # GPU 메모리 제한 설정 (DeepSpeed ZeRO Stage 3 - 8 GPU)
            # GPU 0-7에 대해 메모리 제한 설정
            gpu_memory_fractions = {
                0: 0.90,  # QWEN 메인 프로세스 (약간 낮게)
                1: 0.95,  # QWEN 서브 프로세스
                2: 0.95,  # QWEN 서브 프로세스
                3: 0.95,  # QWEN 서브 프로세스
                4: 0.95,  # CLIP 메인
                5: 0.95,  # CLIP 백업
                6: 0.95,  # SD3 메인
                7: 0.95   # SD3 백업
            }
            
            for gpu_id, fraction in gpu_memory_fractions.items():
                if gpu_id < torch.cuda.device_count():
                    torch.cuda.set_per_process_memory_fraction(fraction, device=gpu_id)
                    logger.info(f"🔧 GPU {gpu_id} 메모리 제한 설정: {int(fraction*100)}% (8 GPU 모드)")
            
            # PyTorch 메모리 할당 최적화
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            logger.info("🧹 초기 GPU 메모리 정리 및 최적화 완료")
        
        # 1. QWEN VL 모델 로드 (Accelerate로 분산)
        logger.info("\n🧠 QWEN VL 모델 + 전체 학습 로딩... (DeepSpeed ZeRO Stage 3)")
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
            logger.info("🎯 메인 프로세스: 통합 모델들 로딩 (GPU 4-7)")
            
            # CLIP 리워드 모델 (GPU 4번 메인)
            reward_model = CLIPReward(device="cuda:4")
            logger.info("✅ CLIP 리워드 모델 로드 완료 (GPU 4)")
            
            # Stable Diffusion 3 파이프라인 (GPU 6번 메인) - 1개만 로딩
            sd_pipeline = load_stable_diffusion_pipeline(device="cuda:6")
            logger.info("✅ SD3 파이프라인 로드 완료 (GPU 6) - 1개만 로딩")
            
            # Reference 모델은 전체 학습에서 비활성화
            logger.info("🎯 전체 학습 모드: Reference 모델 비활성화")
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
            num_epochs=12,  # LoRA로 더 많은 에포크 가능
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
            
            # 9. LoRA 모델 저장 (메인 프로세스에서만)
            logger.info("\n💾 LoRA 모델 저장...")
            save_dir = Path("checkpoints")
            save_dir.mkdir(exist_ok=True)
            
            # LoRA 어댑터 저장
            lora_path = save_dir / "qwen_grpo_lora"
            
            # Accelerate unwrap으로 원본 LoRA 모델 가져오기
            unwrapped_model = accelerator.unwrap_model(qwen_model.model)
            
            # LoRA 어댑터만 저장
            unwrapped_model.save_pretrained(lora_path)
            
            # 메타데이터 저장
            metadata_path = save_dir / "training_metadata.json"
            import json
            metadata = {
                'baseline_reward': avg_baseline,
                'trained_reward': avg_trained,
                'improvement': avg_trained - avg_baseline,
                'lora_config': {
                    'r': 16,
                    'alpha': 32,
                    'dropout': 0.1
                },
                'training_config': {
                    'learning_rate': config.learning_rate,
                    'batch_size': config.batch_size,
                    'num_epochs': 8
                }
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ LoRA 모델 저장 완료: {lora_path}")
            logger.info(f"✅ 메타데이터 저장 완료: {metadata_path}")
            
            # LoRA 파라미터 정보 출력
            lora_info = qwen_model.get_lora_trainable_params()
            logger.info(f"📊 LoRA 학습 파라미터: {lora_info['trainable_params']:,}")
            logger.info(f"📊 전체 파라미터: {lora_info['all_params']:,}")
            logger.info(f"📊 학습 비율: {lora_info['trainable_percentage']:.2f}%")
        
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

            