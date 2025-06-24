# 순수 GRPO VLM 학습 시스템

**Value Network 없이 오직 Policy Network만 사용하는 올바른 GRPO 구현**

## 🎯 개요

이 프로젝트는 **순수 GRPO (Group Relative Policy Optimization)** 알고리즘을 사용하여 VLM(Vision-Language Model)을 학습하는 시스템입니다. easyr1과 동일한 구조로, Value Network 없이 그룹 평균을 implicit baseline으로 사용합니다.

### 주요 특징

- ✅ **순수 GRPO**: Value Network 완전 제거, 오직 Policy Network만 사용
- ✅ **그룹 기반 Advantage**: 프롬프트별 리워드 그룹화 및 정규화
- ✅ **실제 모델 사용**: QWEN VL, Stable Diffusion 3, CLIP
- ✅ **GPU 최적화**: 멀티 GPU 지원 및 메모리 효율성
- ✅ **완전한 파이프라인**: 데이터 로딩부터 모델 저장까지

## 🏗️ 시스템 구조

```
순수 GRPO VLM 학습 시스템 (멀티 GPU 분산)
├── GPU 0: QWEN VL (프롬프트 향상)
├── GPU 1: Stable Diffusion 3 (이미지 생성)
├── GPU 2: CLIP (리워드 계산)
└── 순수 GRPO 트레이너 (Value Network 없음)
```

### 핵심 컴포넌트

1. **PureGRPOPolicy**: 오직 정책 네트워크만 포함
2. **PureGRPOPromptEnvironment**: 토큰 레벨 프롬프트 환경
3. **PureGRPOTrainer**: 순수 GRPO 학습 로직

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# GPU 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### 2. 학습 실행

```bash
# 메인 학습 스크립트 실행
python main.py
```

### 3. SLURM 환경 (선택사항)

```bash
# SLURM 작업 제출
sbatch job.slurm
```

## 📋 설정

### GRPO 하이퍼파라미터

```python
config = PureGRPOConfig(
    learning_rate=1e-6,        # 학습률
    batch_size=4,              # 배치 크기
    num_rollouts=5,            # 그룹별 롤아웃 수
    max_new_tokens=20,         # 최대 생성 토큰
    temperature=1.2,           # 샘플링 온도
    top_k=100,                 # Top-K 필터링
    top_p=0.9,                 # Top-P 필터링
    kl_coef=0.02,             # KL 페널티 계수
    clip_ratio=0.2,           # PPO 클리핑 비율
    entropy_coef=0.01         # 엔트로피 계수
)
```

### 학습 프롬프트

시스템은 다양한 난이도의 프롬프트로 학습합니다:

- 기본 프롬프트 (동물, 풍경 등)
- 도전적인 프롬프트 (색상 조합, 재질 등)
- 복잡한 장면 (군중, 수중, 고대 유적 등)

## 🔧 주요 파일

- **`main.py`**: 메인 학습 스크립트
- **`trainer_grpo_pure.py`**: 순수 GRPO 트레이너 구현
- **`qwen.py`**: QWEN VL 모델 래퍼
- **`clip_reward.py`**: CLIP 리워드 계산기
- **`job.slurm`**: SLURM 작업 스크립트

## 📊 학습 과정

### 1. 모델 로딩

- QWEN VL 7B Instruct 모델
- Stable Diffusion 3 Medium 모델
- CLIP ViT-B/32 모델

### 2. 베이스라인 측정

- 학습 전 성능 측정
- 3개 테스트 프롬프트로 평가

### 3. GRPO 학습

- 프롬프트별 다중 롤아웃 수집
- 그룹 기반 Advantage 계산
- Policy Network만 업데이트

### 4. 성능 평가

- 학습 후 성능 측정
- 개선도 분석 및 로깅

### 5. 모델 저장

- 학습된 Policy Network 저장
- 설정 및 성능 메트릭 저장

## 📈 GRPO 알고리즘

### Advantage 계산 (easyr1과 동일)

```python
# 프롬프트별 리워드 그룹화
for prompt_idx, rewards in prompt_rewards.items():
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards) + 1e-8
    # 그룹 내 정규화 (평균=0, 표준편차=1)
    normalized_rewards = [(r - mean_reward) / std_reward for r in rewards]
```

### 손실 함수 (Value Loss 없음)

```python
total_loss = (
    policy_loss +                    # PPO 정책 손실
    kl_coef * kl_penalty -          # KL 발산 페널티
    entropy_coef * entropy          # 엔트로피 보너스
    # ❌ value_loss 없음!
)
```

## 🎯 핵심 차이점

### 기존 PPO vs 순수 GRPO

| 구분          | 기존 PPO       | 순수 GRPO   |
| ------------- | -------------- | ----------- |
| Value Network | ✅ 있음        | ❌ 없음     |
| Baseline      | Value Network  | 그룹 평균   |
| Advantage     | GAE/TD         | 그룹 정규화 |
| 학습 대상     | Policy + Value | Policy만    |
| 메모리 사용량 | 높음           | 낮음        |

## 📁 출력 파일

- **`grpo_training.log`**: 학습 로그
- **`checkpoints/pure_grpo_policy.pth`**: 저장된 모델
- **`images/`**: 생성된 이미지들 (선택사항)

## 🔍 모니터링

학습 중 다음 메트릭들이 로깅됩니다:

- `total_loss`: 전체 손실
- `policy_loss`: 정책 손실
- `kl_penalty`: KL 발산 페널티
- `entropy`: 정책 엔트로피
- `avg_advantage`: 평균 Advantage
- `learning_rate`: 현재 학습률

## 🚨 주의사항

1. **GPU 요구사항**: 최소 3개의 GPU 필요 (QWEN, SD3, CLIP 분산)
2. **GPU 메모리**: 각 GPU당 최소 16GB VRAM 권장
3. **모델 다운로드**: 첫 실행 시 모델 다운로드로 시간 소요
4. **Hugging Face 토큰**: SD3 사용을 위해 HF 토큰 필요할 수 있음

### GPU 배치 전략

- **GPU 0**: QWEN VL 7B 모델 (프롬프트 향상)
- **GPU 1**: Stable Diffusion 3 Medium (이미지 생성)
- **GPU 2**: CLIP ViT-B/32 (리워드 계산)

이 배치는 각 모델의 메모리 요구사항과 연산 특성을 고려하여 최적화되었습니다.

## 📚 참고자료

- [GRPO 논문](https://arxiv.org/abs/2402.14740)
- [easyr1 구현](https://github.com/openpsi-project/swebench-docker)
- [QWEN VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)

## 🎉 성공 사례

```
🎯 순수 GRPO 학습 결과 (Value Network 없음)
📊 베이스라인 리워드: 8.245
📈 학습 후 리워드: 8.934
🔄 개선도: +0.689
📈 개선률: +8.4%
✅ 학습이 성공적으로 개선되었습니다!
```

---

**순수 GRPO로 더 효율적이고 안정적인 VLM 학습을 경험해보세요!** 🚀
