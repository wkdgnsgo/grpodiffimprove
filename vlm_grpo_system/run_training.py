#!/usr/bin/env python3
"""
VLM GRPO Training Script
========================

VLM GRPO 시스템을 실행하는 간단한 스크립트입니다.

사용법:
    python run_training.py                    # 기본 설정으로 실행
    python run_training.py --config my.json  # 커스텀 설정으로 실행
    python run_training.py --test            # 테스트 모드로 실행

Author: AI Assistant
Date: 2025-01-22
"""

import argparse
import sys
import os
from pathlib import Path

# 현재 디렉토리를 Python path에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="VLM GRPO Training System")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="설정 파일 경로 (기본값: config/default_config.json)"
    )
    
    parser.add_argument(
        "--test", 
        action="store_true",
        help="테스트 모드로 실행 (짧은 학습)"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="디버그 모드로 실행 (상세 로그)"
    )
    
    parser.add_argument(
        "--no-wandb", 
        action="store_true",
        help="Wandb 없이 실행"
    )
    
    args = parser.parse_args()
    
    # 설정 파일 경로 설정
    if args.config is None:
        args.config = current_dir / "config" / "default_config.json"
    
    print("🚀 VLM GRPO Training System")
    print("=" * 50)
    print(f"📁 작업 디렉토리: {current_dir}")
    print(f"⚙️ 설정 파일: {args.config}")
    print(f"🧪 테스트 모드: {args.test}")
    print(f"🐛 디버그 모드: {args.debug}")
    print(f"📈 Wandb 사용: {not args.no_wandb}")
    print("=" * 50)
    
    try:
        # 메인 트레이너 임포트 및 실행
        from integration.main_trainer import VLMGRPOSystem
        
        # 시스템 초기화
        system = VLMGRPOSystem(config_path=str(args.config))
        
        # 테스트 모드 설정 적용
        if args.test:
            print("🧪 테스트 모드: 빠른 학습 설정 적용")
            system.config.update({
                "num_iterations": 5,
                "group_size": 2,
                "validation_interval": 2,
                "checkpoint_interval": 3
            })
        
        # Wandb 설정 적용
        if args.no_wandb:
            print("📈 Wandb 비활성화")
            system.config["use_wandb"] = False
        
        # 디버그 모드 설정
        if args.debug:
            print("🐛 디버그 모드: 상세 로그 활성화")
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
        
        # 컴포넌트 초기화
        print("\n🔧 시스템 컴포넌트 초기화 중...")
        system.initialize_components()
        
        # 학습 실행
        print("\n🚀 학습 시작!")
        system.run_training()
        
        print("\n✅ 학습이 성공적으로 완료되었습니다!")
        print(f"📁 결과 폴더: {system.config['output_dir']}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 학습이 중단되었습니다.")
        sys.exit(1)
        
    except ImportError as e:
        print(f"\n❌ 모듈 임포트 오류: {e}")
        print("\n💡 해결 방법:")
        print("1. 필요한 패키지들이 설치되어 있는지 확인하세요:")
        print("   pip install torch transformers diffusers pillow numpy")
        print("2. 모든 모듈 파일이 올바른 위치에 있는지 확인하세요.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ 학습 중 오류 발생: {e}")
        print("\n💡 문제 해결:")
        print("1. 설정 파일이 올바른지 확인하세요.")
        print("2. 데이터 파일이 존재하는지 확인하세요.")
        print("3. 충분한 메모리와 저장 공간이 있는지 확인하세요.")
        
        if args.debug:
            import traceback
            traceback.print_exc()
        
        sys.exit(1)


def create_sample_data():
    """샘플 데이터 생성 함수"""
    print("📊 샘플 데이터 생성 중...")
    
    try:
        from utils.data_loader import create_sample_data
        create_sample_data(
            train_path="train_prompts.jsonl",
            val_path="val_prompts.jsonl"
        )
        print("✅ 샘플 데이터가 생성되었습니다.")
        
    except Exception as e:
        print(f"❌ 샘플 데이터 생성 실패: {e}")


def show_system_info():
    """시스템 정보 출력"""
    print("💻 시스템 정보:")
    print(f"  - Python: {sys.version}")
    print(f"  - 작업 디렉토리: {os.getcwd()}")
    
    try:
        import torch
        print(f"  - PyTorch: {torch.__version__}")
        print(f"  - CUDA 사용 가능: {torch.cuda.is_available()}")
        print(f"  - MPS 사용 가능: {torch.backends.mps.is_available()}")
    except ImportError:
        print("  - PyTorch: 설치되지 않음")
    
    try:
        import transformers
        print(f"  - Transformers: {transformers.__version__}")
    except ImportError:
        print("  - Transformers: 설치되지 않음")


if __name__ == "__main__":
    # 시스템 정보 출력
    show_system_info()
    print()
    
    # 샘플 데이터가 없으면 생성
    if not Path("train_prompts.jsonl").exists():
        create_sample_data()
        print()
    
    # 메인 실행
    main() 