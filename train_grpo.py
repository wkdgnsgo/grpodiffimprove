"""
GRPO Training Main Script
========================

QWEN 모델을 GRPO 알고리즘으로 학습시키는 메인 스크립트입니다.

실행 예시:
    python train_grpo.py

Author: AI Assistant
Date: 2025-01-22
"""

import sys
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import List

# 로컬 모듈 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Multi-GPU 설정 먼저 초기화
from gpu_config import setup_multi_gpu

from models.qwen_wrapper import QwenWrapper
from models.sd3_generator import SD3Generator
from models.clip_reward import CLIPRewardCalculator
from training.grpo_trainer import GRPOTrainer, GRPOConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('grpo_training.log')
    ]
)

logger = logging.getLogger(__name__)

def get_training_prompts() -> List[str]:
    """학습용 프롬프트 세트 반환 - Challenging cases 포함"""
    return [
        # 기본 프롬프트들
        "a cute cat sitting on a windowsill",
        "a beautiful sunset over mountains",
        "a robot walking in a futuristic city",
        "a delicious pizza with melted cheese",
        "a magical forest with glowing trees",
        
        # SD3가 어려워하는 색상 조합 (Challenging cases)
        "a purple rabbit eating carrots",
        "a green cat with blue eyes",
        "a yellow dog with pink spots",
        "a red elephant with white stripes",
        "a blue horse running in a field",
        "an orange penguin on ice",
        "a pink tiger in the jungle",
        "a silver monkey climbing trees",
        
        # 복잡한 색상과 속성 조합
        "a rainbow colored fish swimming underwater",
        "a transparent glass butterfly",
        "a metallic chrome fox in a forest",
        "a neon glowing wolf at night",
        "a crystal ice bear in a cave",
        "a golden feathered dragon",
        
        # 이상한/모순적인 조합들
        "a square wheel rolling down a hill",
        "a flying fish with wings made of leaves",
        "a tree growing upside down with roots in the sky",
        "a house made of clouds floating in water",
        "a car with legs instead of wheels",
        "a book that is also a bird",
        
        # 복잡한 장면들
        "multiple colored cats playing chess",
        "a purple wizard cat casting green magic",
        "a robot made of different colored fruits",
        "a rainbow bridge connecting two moons",
        "a city where all buildings are different colors",
        "a garden with flowers that are also animals",
        
        # 추상적/개념적 프롬프트들
        "the concept of happiness as a creature",
        "time flowing backwards in a room",
        "music made visible as colorful shapes",
        "a dream within a dream landscape",
        "thoughts becoming physical objects",
        "emotions taking the form of weather"
    ]

def plot_training_results(trainer: GRPOTrainer, save_path: str = "grpo_results.png"):
    """학습 결과 플롯 생성"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. 평균 보상
    axes[0, 0].plot(trainer.iteration_rewards, label='Avg Reward')
    axes[0, 0].set_title('GRPO QWEN: Average Reward per Iteration')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Avg Reward')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    # 이동 평균 추가
    if len(trainer.iteration_rewards) >= 5:
        window = min(5, len(trainer.iteration_rewards))
        ma_rewards = np.convolve(trainer.iteration_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(trainer.iteration_rewards)), ma_rewards, 
                       label=f'{window}-iter MA', linestyle='--')
        axes[0, 0].legend()
    
    # 2. 정책 목적 함수
    axes[0, 1].plot(trainer.iteration_policy_losses, label='Policy Objective')
    axes[0, 1].set_title('GRPO QWEN: Policy Objective per Iteration')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Objective Value')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    # 3. KL 발산
    axes[0, 2].plot(trainer.iteration_kl_divs, label='KL Divergence')
    axes[0, 2].set_title('GRPO QWEN: KL Divergence per Iteration')
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('KL Divergence')
    axes[0, 2].grid(True)
    axes[0, 2].legend()
    
    # 4. 엔트로피
    axes[1, 0].plot(trainer.iteration_entropies, label='Entropy')
    axes[1, 0].set_title('GRPO QWEN: Policy Entropy per Iteration')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Entropy')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    # 5. 보상 분포 히스토그램
    if trainer.iteration_rewards:
        axes[1, 1].hist(trainer.iteration_rewards, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].set_xlabel('Reward Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)
    
    # 6. 학습 통계 요약
    axes[1, 2].axis('off')
    stats_text = f"""
Learning Statistics:
━━━━━━━━━━━━━━━━━━━━
Total Iterations: {len(trainer.iteration_rewards)}
Final Avg Reward: {trainer.iteration_rewards[-1]:.3f if trainer.iteration_rewards else 0:.3f}
Max Reward: {max(trainer.iteration_rewards) if trainer.iteration_rewards else 0:.3f}
Min Reward: {min(trainer.iteration_rewards) if trainer.iteration_rewards else 0:.3f}
Reward Std: {np.std(trainer.iteration_rewards) if trainer.iteration_rewards else 0:.3f}

Final KL Div: {trainer.iteration_kl_divs[-1]:.4f if trainer.iteration_kl_divs else 0:.4f}
Final Entropy: {trainer.iteration_entropies[-1]:.4f if trainer.iteration_entropies else 0:.4f}
    """
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                    fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    logger.info(f"📊 Training results saved to {save_path}")

def main():
    """메인 학습 함수"""
    logger.info("🚀 Starting GRPO Training for QWEN Model")
    
    try:
        # 0. Multi-GPU 환경 설정
        logger.info("🔧 Setting up Multi-GPU environment...")
        gpu_config = setup_multi_gpu()
        
        # 1. 모델 초기화 (각각 다른 GPU에 배치)
        logger.info("📥 Initializing models on assigned GPUs...")
        logger.info("  🎯 QWEN VL → GPU 1 (cuda:0)")
        logger.info("  🖼️ SD3 → GPU 2 (cuda:1)")  
        logger.info("  📏 CLIP → GPU 3 (cuda:2)")
        
        # QWEN 모델 로드 (GPU 1)
        qwen_model = QwenWrapper()
        
        # SD3 Generator 로드 (GPU 2)
        sd3_generator = SD3Generator(
            height=1024,
            width=1024,
            num_inference_steps=28,
            guidance_scale=7.0
        )
        
        # CLIP 보상 계산기 로드 (GPU 3)
        clip_calculator = CLIPRewardCalculator()
        
        logger.info("✅ All models loaded successfully on assigned GPUs")
        
        # GPU 상태 출력
        gpu_config.print_gpu_status()
        
        # 2. GRPO 설정
        config = GRPOConfig(
            learning_rate=1e-5,
            group_size=2,  # 더 작은 배치 (큰 액션 공간 때문에)
            num_iterations=50,  # 더 많은 iteration으로 다양한 프롬프트 학습
            grpo_epochs=3,
            max_new_tokens=12,  # 더 자유로운 생성을 위해 토큰 수 증가
            
            # GRPO 하이퍼파라미터 (큰 액션 공간에 맞게 조정)
            gamma=0.95,  # 약간 낮은 할인 팩터
            grpo_kl_beta=0.05,  # 높은 KL 페널티로 안정성 확보
            grpo_clip_epsilon=0.2,
            entropy_coeff=0.02  # 높은 엔트로피로 탐험 장려
        )
        
        # 3. 트레이너 초기화
        logger.info("🔧 Initializing GRPO trainer...")
        trainer = GRPOTrainer(qwen_model, sd3_generator, clip_calculator, config)
        
        # 4. 학습 및 검증 프롬프트 준비
        all_prompts = get_training_prompts()
        
        # 프롬프트를 학습/검증으로 분할
        np.random.seed(42)  # 재현 가능성
        indices = np.random.permutation(len(all_prompts))
        split_idx = int(0.8 * len(all_prompts))
        
        training_prompts = [all_prompts[i] for i in indices[:split_idx]]
        validation_prompts = [all_prompts[i] for i in indices[split_idx:]]
        
        logger.info(f"📝 Prepared {len(training_prompts)} training prompts")
        logger.info(f"📝 Prepared {len(validation_prompts)} validation prompts")
        
        # Challenging 프롬프트 카테고리 로깅
        challenging_keywords = ["purple", "green cat", "rainbow", "transparent", "square wheel", "upside down", "concept of"]
        challenging_count = sum(1 for prompt in training_prompts if any(keyword in prompt for keyword in challenging_keywords))
        logger.info(f"🎯 Challenging prompts in training set: {challenging_count}/{len(training_prompts)}")
        
        # 5. 학습 루프
        logger.info("🎯 Starting GRPO training loop...")
        
        for iteration in range(config.num_iterations):
            logger.info(f"\n{'='*50}")
            logger.info(f"🔄 Iteration {iteration + 1}/{config.num_iterations}")
            logger.info(f"{'='*50}")
            
            # 현재 iteration의 프롬프트 선택
            np.random.seed(iteration)  # 재현 가능성을 위한 시드
            selected_prompts = np.random.choice(
                training_prompts, 
                size=config.group_size, 
                replace=False
            ).tolist()
            
            logger.info(f"📋 Selected prompts for this iteration:")
            for i, prompt in enumerate(selected_prompts):
                logger.info(f"  {i+1}. {prompt}")
            
            # 학습 iteration 실행
            try:
                results = trainer.train_iteration(selected_prompts)
                
                # 결과 로깅
                logger.info(f"📊 Iteration {iteration + 1} Results:")
                logger.info(f"  Avg Reward: {results['avg_reward']:.4f}")
                logger.info(f"  Avg Length: {results['avg_length']:.1f}")
                logger.info(f"  Policy Obj: {results['policy_objective']:.6f}")
                logger.info(f"  KL Div: {results['kl_divergence']:.6f}")
                logger.info(f"  Entropy: {results['entropy']:.4f}")
                
                # GPU 메모리 정리 (메모리 누수 방지)
                if (iteration + 1) % 5 == 0:
                    gpu_config.clear_gpu_memory()
                    logger.info("🧹 GPU memory cleared")
                
                # 주기적 검증 및 플롯
                if (iteration + 1) % 10 == 0:
                    logger.info(f"🔍 Running validation at iteration {iteration + 1}...")
                    
                    # 검증 실행 (작은 배치로)
                    val_prompts = np.random.choice(validation_prompts, size=min(3, len(validation_prompts)), replace=False).tolist()
                    val_results = trainer.train_iteration(val_prompts)
                    
                    logger.info(f"📈 Validation Results:")
                    logger.info(f"  Val Reward: {val_results['avg_reward']:.4f}")
                    logger.info(f"  Val Length: {val_results['avg_length']:.1f}")
                    
                    # 특별히 challenging 프롬프트들 테스트
                    challenging_prompts = [p for p in validation_prompts if any(keyword in p for keyword in challenging_keywords)]
                    if challenging_prompts:
                        test_prompt = np.random.choice(challenging_prompts)
                        logger.info(f"🎯 Testing challenging prompt: '{test_prompt}'")
                        challenge_results = trainer.train_iteration([test_prompt])
                        logger.info(f"  Challenge Reward: {challenge_results['avg_reward']:.4f}")
                    
                    plot_training_results(trainer, f"grpo_results_iter_{iteration+1}.png")
                
            except Exception as e:
                logger.error(f"❌ Error in iteration {iteration + 1}: {e}")
                continue
        
        # 6. 최종 결과 저장
        logger.info("\n🎉 Training completed successfully!")
        
        # 최종 플롯 생성
        plot_training_results(trainer, "grpo_final_results.png")
        
        # 학습된 모델 저장
        try:
            qwen_model.save_model("./saved_models/qwen_grpo_trained")
            logger.info("💾 Trained model saved to ./saved_models/qwen_grpo_trained")
        except Exception as e:
            logger.warning(f"⚠️ Could not save model: {e}")
        
        # 최종 challenging 프롬프트 테스트
        logger.info(f"\n🎯 Final Challenging Prompts Test:")
        challenging_test_prompts = [
            "a purple rabbit eating carrots",
            "a green cat with blue eyes", 
            "a square wheel rolling down a hill",
            "the concept of happiness as a creature",
            "a transparent glass butterfly"
        ]
        
        for i, prompt in enumerate(challenging_test_prompts):
            logger.info(f"\n🧪 Testing Challenge {i+1}: '{prompt}'")
            try:
                challenge_result = trainer.train_iteration([prompt])
                logger.info(f"  Final Challenge Reward: {challenge_result['avg_reward']:.4f}")
                logger.info(f"  Generated Length: {challenge_result['avg_length']:.1f} tokens")
            except Exception as e:
                logger.warning(f"  Challenge test failed: {e}")
        
        # 최종 통계 출력
        if trainer.iteration_rewards:
            logger.info(f"\n📈 Final Training Statistics:")
            logger.info(f"  Final Reward: {trainer.iteration_rewards[-1]:.4f}")
            logger.info(f"  Best Reward: {max(trainer.iteration_rewards):.4f}")
            logger.info(f"  Reward Improvement: {trainer.iteration_rewards[-1] - trainer.iteration_rewards[0]:.4f}")
            logger.info(f"  Training Stability (Reward Std): {np.std(trainer.iteration_rewards):.4f}")
            
            # Challenging 프롬프트 성능 분석
            recent_rewards = trainer.iteration_rewards[-10:]  # 최근 10개
            logger.info(f"  Recent Performance (last 10 iter): {np.mean(recent_rewards):.4f} ± {np.std(recent_rewards):.4f}")
            
            # 개선 여부 판단
            if len(trainer.iteration_rewards) >= 20:
                early_rewards = trainer.iteration_rewards[:10]
                late_rewards = trainer.iteration_rewards[-10:]
                improvement = np.mean(late_rewards) - np.mean(early_rewards)
                logger.info(f"  Overall Improvement: {improvement:.4f} {'✅' if improvement > 0 else '❌'}")
            
            logger.info(f"\n🎉 Training completed! The model should now generate more creative and reward-optimized prompts.")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Training interrupted by user")
        if 'trainer' in locals():
            plot_training_results(trainer, "grpo_interrupted_results.png")
    
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main() 