"""
GRPO Training Script for QWEN Model
===================================

EasyR1 기반의 새로운 GRPO 트레이너를 사용한 학습 스크립트

Author: AI Assistant
Date: 2025-01-22
"""

import os
import sys
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any
import json

# Multi-GPU 설정 import
from gpu_config import setup_multi_gpu

# 모델 imports
from models.qwen_wrapper import QwenWrapper
from models.sd3_generator import SD3Generator
from models.clip_reward import CLIPRewardCalculator

# 학습 관련 imports
from training.grpo_trainer import GRPOTrainer, GRPOConfig

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

# 도전적인 프롬프트 데이터셋
CHALLENGING_PROMPTS = [
    # 기본 동물/객체
    "a cat sitting on a chair",
    "a beautiful sunset over mountains",
    "a robot playing guitar",
    "a flower garden in spring",
    "an old castle on a hill",
    
    # SD3 어려운 색상 조합
    "a purple rabbit sitting in grass",
    "a green cat with yellow eyes",
    "a blue elephant in the desert",
    "a red bird with black wings",
    "a yellow dog with pink spots",
    
    # 모순적인 개념들
    "a square wheel rolling down a hill",
    "an upside down tree growing in the sky",
    "a transparent fish swimming in air",
    "a silent thunderstorm with visible sound waves",
    "a car with legs instead of wheels",
    
    # 추상적 개념들
    "the concept of happiness visualized as colors",
    "time flowing backwards in a clock",
    "music made visible as geometric shapes",
    "the feeling of nostalgia as a landscape",
    "dreams transforming into reality",
    
    # 복잡한 재질/텍스처
    "a glass sculpture of a dragon",
    "a metallic chrome rose on black velvet",
    "a wooden elephant with crystal eyes",
    "a paper airplane made of liquid mercury",
    "a stone butterfly with feather wings",
    
    # 환상적/초현실적
    "a floating island with waterfalls going upward",
    "a library where books fly like birds",
    "a mirror that shows different seasons",
    "a doorway leading to another dimension",
    "a phoenix made of pure light",
    
    # 고급 조명/분위기
    "a portrait lit by candlelight",
    "neon lights reflecting on wet streets",
    "sunbeams through stained glass windows",
    "aurora borealis over a frozen lake",
    "a lighthouse beam cutting through fog"
]

def create_train_val_split(prompts: List[str], train_ratio: float = 0.8) -> tuple:
    """프롬프트를 train/validation으로 분할"""
    np.random.shuffle(prompts)
    split_idx = int(len(prompts) * train_ratio)
    return prompts[:split_idx], prompts[split_idx:]

def plot_training_progress(trainer: GRPOTrainer, save_path: str = "training_progress.png"):
    """학습 진행 상황을 플롯"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GRPO Training Progress (EasyR1)', fontsize=16, fontweight='bold')
        
        iterations = range(1, len(trainer.iteration_rewards) + 1)
        
        # 보상 그래프
        axes[0, 0].plot(iterations, trainer.iteration_rewards, 'b-', linewidth=2, marker='o')
        axes[0, 0].set_title('Average Reward per Iteration')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Policy Loss 그래프
        if trainer.iteration_policy_losses:
            axes[0, 1].plot(iterations, trainer.iteration_policy_losses, 'r-', linewidth=2, marker='s')
            axes[0, 1].set_title('Policy Loss per Iteration')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
        
        # KL Divergence 그래프
        if trainer.iteration_kl_divs:
            axes[1, 0].plot(iterations, trainer.iteration_kl_divs, 'g-', linewidth=2, marker='^')
            axes[1, 0].set_title('KL Divergence per Iteration')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('KL Divergence')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 보상 히스토그램 (최근 10개 iteration)
        if len(trainer.iteration_rewards) >= 10:
            recent_rewards = trainer.iteration_rewards[-10:]
            axes[1, 1].hist(recent_rewards, bins=min(10, len(recent_rewards)), alpha=0.7, color='purple')
            axes[1, 1].set_title('Recent Reward Distribution (Last 10 Iterations)')
            axes[1, 1].set_xlabel('Reward')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Training progress plot saved to {save_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to create training plot: {e}")

def validate_model(trainer: GRPOTrainer, val_prompts: List[str]) -> Dict[str, float]:
    """검증 세트로 모델 성능 평가"""
    logger.info(f"🔍 Validating on {len(val_prompts)} prompts...")
    
    # 검증 trajectory 수집
    val_data = trainer.collect_group_trajectories(val_prompts)
    
    val_rewards = val_data.get('episode_rewards', [])
    val_lengths = val_data.get('episode_lengths', [])
    
    if not val_rewards:
        return {'val_avg_reward': 0.0, 'val_avg_length': 0.0}
    
    return {
        'val_avg_reward': np.mean(val_rewards),
        'val_avg_length': np.mean(val_lengths),
        'val_std_reward': np.std(val_rewards),
        'val_min_reward': np.min(val_rewards),
        'val_max_reward': np.max(val_rewards)
    }

def save_model_checkpoint(trainer: GRPOTrainer, iteration: int, save_dir: str = "checkpoints"):
    """모델 체크포인트 저장"""
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(save_dir, f"grpo_model_iter_{iteration:03d}.pt")
        
        import torch
        torch.save({
            'iteration': iteration,
            'action_head_state_dict': trainer.action_head.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'kl_coef': trainer.kl_controller.kl_coef,
            'config': trainer.config,
            'iteration_rewards': trainer.iteration_rewards,
            'iteration_policy_losses': trainer.iteration_policy_losses,
            'iteration_kl_divs': trainer.iteration_kl_divs
        }, checkpoint_path)
        
        logger.info(f"💾 Model checkpoint saved to {checkpoint_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save checkpoint: {e}")

def main():
    parser = argparse.ArgumentParser(description='GRPO Training for QWEN Model (EasyR1)')
    parser.add_argument('--num_iterations', type=int, default=50, help='Number of training iterations')
    parser.add_argument('--group_size', type=int, default=4, help='Group size for GRPO')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--kl_type', type=str, default='adaptive', choices=['adaptive', 'fixed'], help='KL controller type')
    parser.add_argument('--max_new_tokens', type=int, default=10, help='Maximum new tokens to generate')
    parser.add_argument('--save_training_data', action='store_true', help='Save training data')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Checkpoint saving interval')
    parser.add_argument('--validation_interval', type=int, default=5, help='Validation interval')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting GRPO Training (EasyR1 Implementation)")
    logger.info(f"Configuration: {vars(args)}")
    
    try:
        # 1. Multi-GPU 환경 설정
        logger.info("🔧 Setting up multi-GPU environment...")
        setup_multi_gpu()
        
        # 2. 모델 초기화
        logger.info("🤖 Initializing models...")
        qwen_model = QwenWrapper()
        sd3_generator = SD3Generator()
        clip_calculator = CLIPRewardCalculator()
        
        # 3. GRPO 설정
        config = GRPOConfig(
            learning_rate=args.learning_rate,
            group_size=args.group_size,
            num_iterations=args.num_iterations,
            max_new_tokens=args.max_new_tokens,
            kl_type=args.kl_type,
            save_training_data=args.save_training_data,
            device="cuda"
        )
        
        logger.info(f"📋 GRPO Config: {config}")
        
        # 4. 트레이너 초기화
        logger.info("🎯 Initializing GRPO trainer...")
        trainer = GRPOTrainer(qwen_model, sd3_generator, clip_calculator, config)
        
        # 5. 데이터셋 분할
        logger.info("📊 Preparing training data...")
        train_prompts, val_prompts = create_train_val_split(CHALLENGING_PROMPTS, train_ratio=0.8)
        
        logger.info(f"📈 Training prompts: {len(train_prompts)}")
        logger.info(f"📉 Validation prompts: {len(val_prompts)}")
        
        # 6. 학습 루프
        logger.info("🔄 Starting training loop...")
        best_val_reward = -float('inf')
        
        for iteration in range(1, args.num_iterations + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 Iteration {iteration}/{args.num_iterations}")
            logger.info(f"{'='*60}")
            
            # 현재 iteration 설정
            trainer.current_iteration = iteration
            
            # 학습 프롬프트 샘플링
            np.random.shuffle(train_prompts)
            current_prompts = train_prompts[:config.group_size]
            
            # 학습 수행
            try:
                results = trainer.train_iteration(current_prompts)
                
                if results:
                    logger.info(f"📊 Training Results:")
                    for key, value in results.items():
                        logger.info(f"  {key}: {value:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Training failed at iteration {iteration}: {e}")
                continue
            
            # 검증 수행
            if iteration % args.validation_interval == 0:
                val_results = validate_model(trainer, val_prompts)
                logger.info(f"🔍 Validation Results:")
                for key, value in val_results.items():
                    logger.info(f"  {key}: {value:.4f}")
                
                # 최고 성능 모델 저장
                current_val_reward = val_results.get('val_avg_reward', 0.0)
                if current_val_reward > best_val_reward:
                    best_val_reward = current_val_reward
                    save_model_checkpoint(trainer, iteration, "best_models")
                    logger.info(f"🏆 New best validation reward: {best_val_reward:.4f}")
            
            # 체크포인트 저장
            if iteration % args.checkpoint_interval == 0:
                save_model_checkpoint(trainer, iteration)
            
            # 진행 상황 플롯
            if iteration % 5 == 0:
                plot_training_progress(trainer, f"training_progress_iter_{iteration:03d}.png")
            
            # GPU 메모리 정리
            if iteration % 10 == 0:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("🧹 GPU memory cleared")
                except:
                    pass
        
        # 최종 결과
        logger.info(f"\n{'='*60}")
        logger.info("🎉 Training completed!")
        logger.info(f"{'='*60}")
        
        if trainer.iteration_rewards:
            logger.info(f"📊 Final Results:")
            logger.info(f"  Best Reward: {max(trainer.iteration_rewards):.4f}")
            logger.info(f"  Final Reward: {trainer.iteration_rewards[-1]:.4f}")
            logger.info(f"  Average Reward: {np.mean(trainer.iteration_rewards):.4f}")
        
        # 최종 플롯 저장
        plot_training_progress(trainer, "final_training_progress.png")
        
        # 최종 모델 저장
        save_model_checkpoint(trainer, args.num_iterations, "final_models")
        
        logger.info("✅ All training tasks completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 