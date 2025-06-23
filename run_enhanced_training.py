#!/usr/bin/env python3
"""
Enhanced VLM GRPO Training Script
=================================

MS Swift CoZ GRPO를 참조하여 개선된 VLM GRPO 학습 스크립트입니다.
LoRA 버전과 전체 학습 버전을 모두 지원합니다.

사용법:
    # LoRA 학습 (메모리 효율적)
    python run_enhanced_training.py --train_type lora --lora_rank 8 --lora_alpha 32
    
    # 전체 학습 (고성능)
    python run_enhanced_training.py --train_type full --use_deepspeed
    
    # QLoRA 학습 (양자화 + LoRA)
    python run_enhanced_training.py --train_type qlora --lora_rank 16

MS Swift 호환 옵션:
    --train_type {full,lora,qlora}
    --lora_rank 8
    --lora_alpha 32
    --target_modules all-linear
    --learning_rate 1e-5
    --per_device_train_batch_size 4
    --num_generations 4
    --temperature 0.9
    --deepspeed zero2
    --use_vllm
    --log_completions

Author: AI Assistant (Based on MS Swift CoZ GRPO)
Date: 2025-01-22
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# 현재 디렉토리를 Python path에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def setup_logging(debug: bool = False):
    """로깅 설정"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('vlm_grpo_training.log')
        ]
    )

def create_ms_swift_style_args():
    """MS Swift 스타일 명령행 인자 생성"""
    parser = argparse.ArgumentParser(
        description="Enhanced VLM GRPO Training (MS Swift Style)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 기본 모델 설정
    parser.add_argument(
        "--model", "--vlm_model",
        type=str,
        default="microsoft/DialoGPT-medium",
        help="VLM 모델 이름"
    )
    
    parser.add_argument(
        "--sd_model",
        type=str,
        default="stabilityai/stable-diffusion-3-medium",
        help="Stable Diffusion 모델 이름"
    )
    
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP 모델 이름"
    )
    
    # 학습 타입 설정 (MS Swift 스타일)
    parser.add_argument(
        "--train_type",
        type=str,
        choices=["full", "lora", "qlora"],
        default="lora",
        help="학습 타입: full(전체), lora(LoRA), qlora(QLoRA)"
    )
    
    parser.add_argument(
        "--torch_dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="모델 데이터 타입"
    )
    
    # LoRA 설정 (MS Swift 기본값)
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank (낮을수록 메모리 효율적)"
    )
    
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (학습 강도 조절)"
    )
    
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout 비율"
    )
    
    parser.add_argument(
        "--target_modules",
        type=str,
        default="all-linear",
        help="LoRA 적용 모듈 (all-linear 또는 특정 모듈명)"
    )
    
    # 학습 하이퍼파라미터
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="학습률"
    )
    
    parser.add_argument(
        "--num_train_epochs", "--num_iterations",
        type=int,
        default=20,
        help="학습 반복 횟수"
    )
    
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="디바이스당 배치 크기"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="그래디언트 누적 스텝"
    )
    
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="그래디언트 클리핑 임계값"
    )
    
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="웜업 비율"
    )
    
    # GRPO 특화 설정
    parser.add_argument(
        "--group_size",
        type=int,
        default=4,
        help="GRPO 그룹 크기"
    )
    
    parser.add_argument(
        "--num_generations",
        type=int,
        default=4,
        help="생성할 후보 수"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="생성 온도"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p 샘플링"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k 샘플링"
    )
    
    # 데이터 설정
    parser.add_argument(
        "--dataset", "--train_data_path",
        type=str,
        default="train_prompts.jsonl",
        help="학습 데이터 경로"
    )
    
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="val_prompts.jsonl",
        help="검증 데이터 경로"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="최대 시퀀스 길이"
    )
    
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=1024,
        help="최대 생성 길이"
    )
    
    # 평가 및 저장 설정
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="평가 주기"
    )
    
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="저장 주기"
    )
    
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="저장할 체크포인트 수"
    )
    
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=5,
        help="로깅 주기"
    )
    
    parser.add_argument(
        "--validation_interval",
        type=int,
        default=5,
        help="검증 주기"
    )
    
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10,
        help="체크포인트 저장 주기"
    )
    
    # 출력 설정
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vlm_grpo_results_enhanced",
        help="출력 디렉토리"
    )
    
    parser.add_argument(
        "--log_completions",
        action="store_true",
        help="생성 결과 로깅"
    )
    
    # 분산 학습 설정
    parser.add_argument(
        "--deepspeed",
        type=str,
        choices=["zero2", "zero3"],
        help="DeepSpeed 설정"
    )
    
    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="DeepSpeed 사용"
    )
    
    # vLLM 설정
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="vLLM 가속 사용"
    )
    
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.7,
        help="vLLM GPU 메모리 사용률"
    )
    
    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=4096,
        help="vLLM 최대 모델 길이"
    )
    
    # 하드웨어 최적화
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="디바이스 설정 (auto/cuda/mps/cpu)"
    )
    
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        default=True,
        help="Flash Attention 사용"
    )
    
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="그래디언트 체크포인팅 사용"
    )
    
    # 실험 추적
    parser.add_argument(
        "--use_wandb", "--report_to",
        action="store_true",
        help="Wandb 사용"
    )
    
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="vlm-grpo-enhanced",
        help="Wandb 프로젝트명"
    )
    
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        help="Wandb 실행명"
    )
    
    # 보상 함수 설정
    parser.add_argument(
        "--reward_funcs",
        type=str,
        nargs="+",
        default=["clip_similarity", "image_quality"],
        help="보상 함수 목록"
    )
    
    # 시스템 프롬프트
    parser.add_argument(
        "--system",
        type=str,
        help="시스템 프롬프트 파일 경로"
    )
    
    # 실행 모드
    parser.add_argument(
        "--test",
        action="store_true",
        help="테스트 모드 (빠른 실행)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 모드"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="설정 파일 경로"
    )
    
    return parser

def apply_test_mode_settings(args):
    """테스트 모드 설정 적용"""
    if args.test:
        print("🧪 Test mode enabled - applying fast settings")
        args.num_train_epochs = 3
        args.group_size = 2
        args.validation_interval = 2
        args.checkpoint_interval = 2
        args.logging_steps = 1
        args.per_device_train_batch_size = min(args.per_device_train_batch_size, 2)

def create_config_from_args(args) -> Dict[str, Any]:
    """명령행 인자에서 설정 딕셔너리 생성"""
    config = {
        # 기본 모델 설정
        "vlm_model": args.model,
        "sd_model": args.sd_model,
        "clip_model": args.clip_model,
        
        # 학습 타입 설정
        "train_type": args.train_type,
        "torch_dtype": args.torch_dtype,
        
        # LoRA 설정
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
        
        # 학습 하이퍼파라미터
        "learning_rate": args.learning_rate,
        "num_iterations": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_grad_norm": args.max_grad_norm,
        "warmup_ratio": args.warmup_ratio,
        
        # GRPO 특화 설정
        "group_size": args.group_size,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "grpo_epochs": 2,
        "kl_beta": 0.01,
        "clip_epsilon": 0.2,
        
        # 데이터 설정
        "train_data_path": args.dataset,
        "val_data_path": args.val_data_path,
        "max_length": args.max_length,
        "max_completion_length": args.max_completion_length,
        
        # 평가 및 저장 설정
        "validation_interval": args.validation_interval,
        "checkpoint_interval": args.checkpoint_interval,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "logging_steps": args.logging_steps,
        
        # 출력 설정
        "output_dir": args.output_dir,
        "log_completions": args.log_completions,
        
        # 분산 학습 설정
        "use_deepspeed": args.use_deepspeed or bool(args.deepspeed),
        "deepspeed_config": args.deepspeed or "zero2",
        
        # 하드웨어 최적화
        "device": args.device,
        "use_flash_attention": args.use_flash_attention,
        "gradient_checkpointing": args.gradient_checkpointing,
        
        # 보상 함수 설정
        "reward_weights": {
            "clip_similarity": 0.6,
            "image_quality": 0.3,
            "semantic_consistency": 0.1
        },
        
        # 실험 추적
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
        
        # 시스템 프롬프트
        "system_prompt": args.system
    }
    
    return config

def print_training_info(args):
    """학습 정보 출력"""
    print("🚀 Enhanced VLM GRPO Training")
    print("=" * 60)
    print(f"📋 Configuration:")
    print(f"   - Train Type: {args.train_type}")
    print(f"   - Model: {args.model}")
    print(f"   - Device: {args.device}")
    print(f"   - Torch Dtype: {args.torch_dtype}")
    
    if args.train_type in ["lora", "qlora"]:
        print(f"🎯 LoRA Settings:")
        print(f"   - Rank: {args.lora_rank}")
        print(f"   - Alpha: {args.lora_alpha}")
        print(f"   - Target Modules: {args.target_modules}")
    
    print(f"⚙️ Training Settings:")
    print(f"   - Learning Rate: {args.learning_rate}")
    print(f"   - Batch Size: {args.per_device_train_batch_size}")
    print(f"   - Iterations: {args.num_train_epochs}")
    print(f"   - Group Size: {args.group_size}")
    print(f"   - Generations: {args.num_generations}")
    
    if args.use_deepspeed or args.deepspeed:
        print(f"⚡ DeepSpeed: {args.deepspeed or 'zero2'}")
    
    if args.use_vllm:
        print(f"🚄 vLLM: Enabled")
    
    print(f"📊 Output:")
    print(f"   - Directory: {args.output_dir}")
    print(f"   - Wandb: {args.use_wandb}")
    print(f"   - Test Mode: {args.test}")
    print("=" * 60)

def check_dependencies():
    """필수 의존성 확인"""
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing_deps.append("transformers")
    
    try:
        import diffusers
    except ImportError:
        missing_deps.append("diffusers")
    
    try:
        from PIL import Image
    except ImportError:
        missing_deps.append("pillow")
    
    if missing_deps:
        print("❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n💡 Install with:")
        print(f"   pip install {' '.join(missing_deps)}")
        return False
    
    return True

def create_sample_data_if_needed(train_path: str, val_path: str):
    """필요시 샘플 데이터 생성"""
    if not Path(train_path).exists() or not Path(val_path).exists():
        print("📊 Creating sample data...")
        try:
            from utils.data_loader import create_sample_data
            create_sample_data(train_path, val_path)
            print("✅ Sample data created")
        except Exception as e:
            print(f"⚠️ Could not create sample data: {e}")

def main():
    """메인 실행 함수"""
    # 명령행 인자 파싱
    parser = create_ms_swift_style_args()
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(args.debug)
    
    # 테스트 모드 설정 적용
    apply_test_mode_settings(args)
    
    # 학습 정보 출력
    print_training_info(args)
    
    # 의존성 확인
    if not check_dependencies():
        sys.exit(1)
    
    # 샘플 데이터 생성
    create_sample_data_if_needed(args.dataset, args.val_data_path)
    
    try:
        # 설정 생성
        config = create_config_from_args(args)
        
        # 설정 파일에서 로드 (있는 경우)
        if args.config and Path(args.config).exists():
            with open(args.config, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            config.update(file_config)
        
        # Enhanced VLM GRPO System 임포트 및 실행
        from integration.main_trainer_enhanced import EnhancedVLMGRPOSystem
        
        # 시스템 초기화
        system = EnhancedVLMGRPOSystem(**config)
        
        # 컴포넌트 초기화
        print("\n🔧 Initializing components...")
        system.initialize_components()
        
        # 시스템 요약 출력
        system.print_system_summary()
        
        # 학습 실행
        print("\n🚀 Starting training...")
        system.run_training()
        
        # 최종 통계 출력
        final_stats = system.get_training_stats()
        print("\n🎉 Training completed!")
        print(f"   - Total samples: {final_stats['total_samples']}")
        print(f"   - Best reward: {final_stats['best_reward']:.4f}")
        print(f"   - Training time: {final_stats['training_time']:.2f}s")
        print(f"   - Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 