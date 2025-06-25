#!/bin/bash

#SBATCH --job-name=qwen-grpo-accelerate # QWEN GRPO Accelerate 멀티 GPU 학습
#SBATCH --nodes=1
#SBATCH --nodelist=hpc-pr-a-pod20
#SBATCH --ntasks-per-node=1 # total number of tasks per node
#SBATCH --cpus-per-task=8 # cpu-cores per task (멀티 GPU + 추론 모델 처리용)
#SBATCH --gres=gpu:6 # 6개 GPU 필요 (QWEN 전체학습: 2-5, CLIP: 6, SD3: 7)
#SBATCH --time=10:00:00 # total run time limit (HH:MM:SS) - 멀티 GPU로 더 긴 학습
#SBATCH --output=qwen_grpo_accelerate_training.out
#SBATCH --error=qwen_grpo_accelerate_training.err
#SBATCH --mail-type=begin # send email when job begins
#SBATCH --mail-type=end # send email when job ends
#SBATCH --mail-user=sunny17@kaist.ac.kr

# Accelerate 멀티 GPU 환경 변수 설정
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true # WandB 비활성화 (선택사항)

echo "WORLD_SIZE="$WORLD_SIZE
echo "CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# GPU 정보 출력
echo "=== GPU 정보 ==="
nvidia-smi
echo ""

# GPU 배치 정보 출력 (DeepSpeed ZeRO Stage 3 전체 학습)
echo "=== GPU 배치 계획 (DeepSpeed ZeRO Stage 3 전체 학습) ==="
echo "GPU 2-5: QWEN 전체 학습 (DeepSpeed ZeRO Stage 3 분산)"
echo "  - GPU 2: QWEN 메인 프로세스 (1/4)"
echo "  - GPU 3: QWEN 서브 프로세스 (2/4)"
echo "  - GPU 4: QWEN 서브 프로세스 (3/4)"
echo "  - GPU 5: QWEN 서브 프로세스 (4/4)"
echo "GPU 6: CLIP 리워드 모델"
echo "  - CLIP 리워드 계산 전용"
echo "GPU 7: Stable Diffusion 3"
echo "  - SD3 파이프라인 (이미지 생성 전용)"
echo ""

# Accelerate 설정 자동 생성
echo "=== Accelerate 설정 생성 ==="
mkdir -p ~/.cache/huggingface/accelerate

cat > ~/.cache/huggingface/accelerate/default_config.yaml << EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: '2,3,4,5'
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

echo "✅ Accelerate 설정 파일 생성 완료"
cat ~/.cache/huggingface/accelerate/default_config.yaml
echo ""

# 메모리 사용량 모니터링 함수
monitor_memory() {
    while true; do
        echo "=== GPU 메모리 사용량 $(date) ==="
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
        echo ""
        sleep 300 # 5분마다 체크
    done
}

# 백그라운드에서 메모리 모니터링 시작
monitor_memory &
MONITOR_PID=$!

# QWEN GRPO Accelerate 멀티 GPU 학습 실행
echo "=== QWEN GRPO Accelerate 멀티 GPU 학습 시작 ==="
echo "시작 시간: $(date)"
echo ""

# 옵션 1: 직접 accelerate 명령어 사용
accelerate launch \
    --config_file ~/.cache/huggingface/accelerate/default_config.yaml \
    --main_process_port $MASTER_PORT \
    main.py

# 옵션 2: Python 스크립트 사용 (위 명령어가 실패할 경우)
# python run_accelerate_training.py

echo ""
echo "완료 시간: $(date)"
echo "=== QWEN GRPO Accelerate 멀티 GPU 학습 완료 ==="

# 메모리 모니터링 종료
kill $MONITOR_PID 2>/dev/null

# 최종 GPU 상태 출력
echo "=== 최종 GPU 상태 ==="
nvidia-smi
echo ""

# 결과 파일 확인
echo "=== 생성된 결과 파일 ==="
ls -la qwen_grpo_results/ 2>/dev/null || echo "결과 디렉토리가 없습니다."
ls -la checkpoints/ 2>/dev/null || echo "체크포인트 디렉토리가 없습니다."
echo ""

