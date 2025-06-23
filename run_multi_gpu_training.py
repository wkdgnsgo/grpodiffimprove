"""
Multi-GPU GRPO Training Runner
=============================

GPU 1, 2, 3번을 사용하여 QWEN 모델을 GRPO로 학습하는 Python 실행 스크립트입니다.

사용법:
    python run_multi_gpu_training.py

Author: AI Assistant  
Date: 2025-01-22
"""

import os
import sys
import subprocess
import logging
import time
from typing import Optional, List

def setup_gpu_environment():
    """GPU 환경 변수 설정"""
    print("🚀 Starting Multi-GPU GRPO Training")
    print("=" * 50)
    
    # GPU 환경 변수 설정
    gpu_env_vars = {
        'CUDA_VISIBLE_DEVICES': '1,2,3',
        'CUDA_LAUNCH_BLOCKING': '0',
        'TORCH_CUDA_ARCH_LIST': '8.0;8.6;8.9;9.0',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
        'NCCL_DEBUG': 'INFO',
        'NCCL_IB_DISABLE': '1',
        # Distributed training 환경 변수 (단일 노드)
        'RANK': '0',
        'WORLD_SIZE': '1',
        'LOCAL_RANK': '0',
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '12355',
        # Hugging Face distributed 비활성화
        'HF_HUB_DISABLE_PROGRESS_BARS': '1',
        'TRANSFORMERS_NO_ADVISORY_WARNINGS': '1',
        'TOKENIZERS_PARALLELISM': 'false'
    }
    
    # 환경 변수 적용
    for key, value in gpu_env_vars.items():
        os.environ[key] = value
    
    print("🔧 GPU Environment Variables:")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"  PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    
    return gpu_env_vars

def check_gpu_status(gpu_ids: List[int] = [1, 2, 3]) -> bool:
    """GPU 상태 확인"""
    print("\n📱 GPU Status Check:")
    
    try:
        # nvidia-smi 명령어 실행
        cmd = [
            "nvidia-smi", 
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_found = False
            
            for line in lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 5:
                        gpu_index = int(parts[0].strip())
                        if gpu_index in gpu_ids:
                            gpu_found = True
                            gpu_name = parts[1].strip()
                            memory_used = parts[2].strip()
                            memory_total = parts[3].strip()
                            utilization = parts[4].strip()
                            
                            print(f"  GPU {gpu_index}: {gpu_name}")
                            print(f"    Memory: {memory_used}MB / {memory_total}MB")
                            print(f"    Utilization: {utilization}%")
            
            return gpu_found
        else:
            print(f"  ⚠️ nvidia-smi failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("  ⚠️ nvidia-smi timeout")
        return False
    except FileNotFoundError:
        print("  ⚠️ nvidia-smi not found")
        return False
    except Exception as e:
        print(f"  ⚠️ GPU check failed: {e}")
        return False

def print_model_assignment():
    """모델 GPU 할당 정보 출력"""
    print("\n🎯 Model GPU Assignment:")
    print("  GPU 1 (cuda:0): QWEN VL Model (Policy Network)")
    print("  GPU 2 (cuda:1): Stable Diffusion 3 (Environment)")
    print("  GPU 3 (cuda:2): CLIP Model (Reward Calculator)")

def run_training():
    """GRPO 학습 실행"""
    print("\n⚡ Starting GRPO Training...")
    print("=" * 50)
    
    try:
        # 현재 스크립트와 같은 디렉토리에서 train_grpo.py 실행
        current_dir = os.path.dirname(os.path.abspath(__file__))
        train_script = os.path.join(current_dir, 'train_grpo.py')
        
        if not os.path.exists(train_script):
            print(f"❌ Training script not found: {train_script}")
            return False
        
        # Python 인터프리터로 학습 스크립트 실행
        cmd = [sys.executable, train_script]
        
        print(f"🐍 Executing: {' '.join(cmd)}")
        print("📝 Training output:")
        print("-" * 50)
        
        # 실시간 출력을 위해 subprocess.Popen 사용
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 실시간 출력
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # 프로세스 완료 대기
        return_code = process.poll()
        
        if return_code == 0:
            print("\n🎉 Training completed successfully!")
            return True
        else:
            print(f"\n❌ Training failed with return code: {return_code}")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        if 'process' in locals():
            process.terminate()
        return False
    except Exception as e:
        print(f"\n❌ Training execution failed: {e}")
        return False

def post_training_gpu_check(gpu_ids: List[int] = [1, 2, 3]):
    """학습 완료 후 GPU 상태 확인"""
    print("\n📊 Post-Training GPU Status:")
    check_gpu_status(gpu_ids)

def main():
    """메인 실행 함수"""
    start_time = time.time()
    
    try:
        # 1. GPU 환경 설정
        gpu_env_vars = setup_gpu_environment()
        
        # 2. GPU 상태 확인
        gpu_available = check_gpu_status([1, 2, 3])
        
        if not gpu_available:
            print("\n⚠️ Warning: Target GPUs (1, 2, 3) not detected")
            response = input("Continue anyway? (y/n): ").lower().strip()
            if response not in ['y', 'yes']:
                print("❌ Training cancelled")
                return False
        
        # 3. 모델 할당 정보 출력
        print_model_assignment()
        
        # 4. 학습 실행
        training_success = run_training()
        
        # 5. 학습 완료 후 GPU 상태 확인
        if training_success:
            post_training_gpu_check([1, 2, 3])
            
            # 실행 시간 계산
            end_time = time.time()
            duration = end_time - start_time
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            
            print(f"\n⏱️ Total Training Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
            print("🎉 Multi-GPU GRPO Training Completed Successfully!")
            return True
        else:
            print("\n❌ Training failed")
            return False
    
    except KeyboardInterrupt:
        print("\n⏹️ Multi-GPU training interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Multi-GPU training failed: {e}")
        return False

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    success = main()
    sys.exit(0 if success else 1) 