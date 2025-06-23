# GRPO for QWEN VL Model 🚀

**QWEN VL 모델을 GRPO 알고리즘으로 학습시키는 시스템**

CartPole GRPO 구현을 기반으로 QWEN VL 모델을 강화학습으로 학습시켜 더 나은 프롬프트 향상 능력을 획득하는 시스템입니다.

## 🎯 GRPO 학습 목적

```
User Prompt → [QWEN VL Policy] → Action Selection → Enhanced Prompt → [SD3 Image] → [CLIP Reward] → Policy Update
```

**핵심 아이디어**:

- **무제한 어휘**: 품질 토큰에 제한되지 않고 전체 어휘에서 자유롭게 선택
- **Challenging Cases**: SD3가 어려워하는 색상 조합, 모순적 개념들도 학습
- **창의적 프롬프트**: 이상한/추상적 프롬프트에서도 높은 보상을 얻는 모델 학습

## 🚀 Challenging 프롬프트 예시

**SD3가 어려워하는 케이스들:**

- `"a purple rabbit eating carrots"` - 비현실적 색상 조합
- `"a green cat with blue eyes"` - 기존 상식과 다른 색상
- `"a square wheel rolling down a hill"` - 모순적/불가능한 조합
- `"the concept of happiness as a creature"` - 추상적 개념의 구현
- `"a transparent glass butterfly"` - 복잡한 재질과 형태

**GRPO 학습 목표**: 이런 어려운 프롬프트들에서도 높은 CLIP 보상을 얻을 수 있는 창의적인 프롬프트 생성

## 📁 프로젝트 구조

```
grpodiffimprove/
├── models/
│   ├── qwen_wrapper.py        # QWEN VL 래퍼 클래스
│   └── clip_reward.py         # CLIP 보상 계산기
├── test_qwen.py               # QWEN 테스트 스크립트
├── test_clip_reward.py        # CLIP 보상 테스트 스크립트
├── test_integrated_system.py  # 통합 시스템 테스트
├── requirements.txt           # 의존성 패키지
└── README.md                  # 이 파일
```

## 🚀 설치 및 사용법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. Multi-GPU GRPO 학습 실행 (권장)

```bash
# GPU 1, 2, 3번을 사용한 최적화된 학습
./run_multi_gpu_training.sh
```

### 3. 단일 GPU 학습 실행

```bash
python train_grpo.py
```

### 4. 테스트 실행

```bash
# 전체 시스템 테스트
python test_grpo_system.py

# QWEN VL 프롬프트 향상 테스트
python test_qwen.py

# CLIP 보상 시스템 테스트
python test_clip_reward.py

# 통합 시스템 테스트 (QWEN VL + CLIP)
python test_integrated_system.py
```

## 🧪 테스트 결과 예시

### 통합 시스템 테스트

```
🏃‍♂️ Quick Integration Test
--------------------------------------------------

📝 User: "cat"
✨ Enhanced: "a beautiful fluffy cat sitting gracefully, high quality, det..."
🎯 Dummy Reward: 0.7680
📊 Quality: 5.00/5.0

📝 User: "sunset"
✨ Enhanced: "stunning sunset over the ocean, golden hour lighting, cinema..."
🎯 Dummy Reward: 0.7380
📊 Quality: 5.00/5.0

📝 User: "robot"
✨ Enhanced: "futuristic robot with metallic finish, sci-fi environment, d..."
🎯 Dummy Reward: 0.8960
📊 Quality: 4.33/5.0

✅ System pipeline working correctly!
Key insight: Enhanced prompt generates image, but CLIP reward uses original user prompt!
```

### QWEN VL 프롬프트 향상

```
🚀 QWEN VL Prompt Enhancement Test
==================================================
📥 Loading QWEN VL model...
✅ Model loaded: Qwen/Qwen2-VL-7B-Instruct
🖥️  Device: mps

🧪 Testing prompt enhancement...
--------------------------------------------------

[Test 1/8]
📝 Original: a cat
✨ Enhanced: a beautiful fluffy orange tabby cat sitting gracefully on a windowsill, soft natural lighting, professional pet photography, high quality, detailed fur texture, 4k resolution
📏 Length: 5 -> 147
🎯 Quality Score: 4.85/5.0
```

## 🔧 주요 기능

### 1. QwenWrapper 클래스

- **모델**: Qwen/Qwen2-VL-7B-Instruct (VL 모델 사용)
- **자동 디바이스 선택**: CUDA > MPS > CPU
- **템플릿 기반 프롬프트**: System + User 메시지 구조
- **후처리**: 불필요한 텍스트 제거 및 정제

```python
from models.qwen_wrapper import QwenWrapper

# 모델 초기화
qwen = QwenWrapper()

# 단일 프롬프트 향상
result = qwen.enhance_prompt("a cat")
print(result['enhanced_prompt'])

# 배치 처리
results = qwen.enhance_prompts_batch(["cat", "dog", "bird"])
```

### 2. CLIPRewardCalculator 클래스

- **모델**: openai/clip-vit-base-patch32
- **보상 범위**: 0.0 ~ 1.0 (1.0에 가까울수록 높은 유사도)
- **핵심**: Enhanced prompt가 아닌 **원본 user prompt**와 이미지 비교

```python
from models.clip_reward import CLIPRewardCalculator

# CLIP 보상 계산기 초기화
clip = CLIPRewardCalculator()

# 보상 계산 (원본 user prompt 사용!)
reward = clip.calculate_reward("a cat", generated_image)
print(f"Reward: {reward:.4f}")  # 0.0~1.0 범위

# 배치 처리
rewards = clip.calculate_rewards_batch(user_prompts, images)
```

## 📊 품질 평가 시스템

테스트 스크립트는 다음 기준으로 향상 품질을 평가합니다:

1. **길이 증가** (1점): 더 상세한 설명
2. **원본 키워드 포함** (1점): 원본 의도 유지
3. **새로운 디테일 추가** (1점): 5개 이상의 새 단어
4. **품질 키워드** (1점): 'high quality', 'detailed', '4k' 등
5. **문장 구조** (1점): 5단어 이상의 의미있는 문장

**총 5점 만점**으로 평가됩니다.

## ⚙️ 설정 옵션

```python
qwen = QwenWrapper(
    model_name="Qwen/Qwen2.5-7B-Instruct",  # 모델 이름
    device="auto",                           # 디바이스 (auto/cuda/mps/cpu)
    max_new_tokens=100,                      # 최대 생성 토큰
    temperature=0.7                          # 생성 온도
)
```

## 🎨 프롬프트 템플릿

시스템 프롬프트는 이미지 생성에 최적화되어 있습니다:

- 원본 주제 유지
- 예술적 스타일 및 조명 추가
- 기술적 명세 포함 (해상도, 품질)
- 간결하면서도 상세한 설명

## 📈 다음 단계

이 기본 시스템을 바탕으로 다음 기능들을 추가할 예정입니다:

- [ ] Stable Diffusion 이미지 생성 연동
- [ ] CLIP 기반 품질 평가
- [ ] 강화학습 (GRPO) 통합
- [ ] 웹 인터페이스 구축
- [ ] 배치 처리 최적화

## 🖥️ Multi-GPU 시스템

### GPU 분배 전략

- **GPU 1 (cuda:0)**: QWEN VL 모델 (정책 네트워크)
- **GPU 2 (cuda:1)**: Stable Diffusion 3 (이미지 생성 환경)
- **GPU 3 (cuda:2)**: CLIP 모델 (보상 계산)

### 환경 변수 자동 설정

```bash
CUDA_VISIBLE_DEVICES=1,2,3
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
NCCL_DEBUG=INFO
```

### 메모리 최적화

- 각 GPU에 85% 메모리 한도 설정
- 주기적 GPU 메모리 정리 (5 iteration마다)
- 모델별 독립적 메모리 관리

## 🛠️ 기술 스택

- **Language Model**: Qwen2-VL-7B-Instruct (VL 모델)
- **Image Generation**: Stable Diffusion 3 Medium
- **Reward Model**: CLIP ViT-B/32
- **Framework**: PyTorch + Transformers + Diffusers
- **Multi-GPU**: 3 GPU 분산 처리
- **Device Support**: CUDA Multi-GPU, Apple Silicon MPS, CPU
- **Python**: 3.8+

---

**Author**: AI Assistant  
**Date**: 2025-01-22  
**Status**: 기본 기능 구현 완료 ✅
