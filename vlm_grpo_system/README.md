# VLM GRPO System 🚀

**Vision Language Model + Group Relative Policy Optimization**

VLM을 사용하여 사용자 프롬프트를 개선하고, Stable Diffusion으로 이미지를 생성하며, CLIP으로 보상을 계산하는 강화학습 시스템입니다.

## 시스템 구조 📋

```
User Prompt → VLM → Enhanced Prompt → SD3 → Image → CLIP Reward → GRPO Update
```

### 폴더 구조

```
vlm_grpo_system/
├── models/                 # 핵심 모델들
│   ├── vlm_wrapper.py     # VLM 프롬프트 개선
│   ├── sd_generator.py    # Stable Diffusion 이미지 생성
│   └── clip_reward.py     # CLIP 보상 계산
├── training/              # 학습 알고리즘
│   └── grpo_trainer.py    # GRPO 트레이너
├── utils/                 # 유틸리티
│   └── data_loader.py     # 데이터 로딩
├── evaluation/            # 평가 시스템
│   └── validator.py       # 검증 평가기
├── integration/           # 통합 시스템
│   ├── main_trainer.py    # 메인 트레이너
│   └── wandb_logger.py    # Wandb 로거
├── config/                # 설정 파일
│   └── default_config.json
└── data/                  # 데이터 파일
```

## 주요 기능 ✨

### 1. Models 폴더 - 핵심 모델들

#### VLM Wrapper (`models/vlm_wrapper.py`)

- **목적**: 사용자의 간단한 프롬프트를 상세한 프롬프트로 개선
- **입력**: "a cat"
- **출력**: "a fluffy orange tabby cat sitting gracefully on a windowsill, soft natural lighting, professional pet photography"
- **기능**:
  - 프롬프트 템플릿 적용
  - 텍스트 생성 파라미터 관리
  - 배치 처리 지원
  - 실패 시 fallback 처리

#### SD3 Generator (`models/sd_generator.py`)

- **목적**: 개선된 프롬프트로 고품질 이미지 생성
- **기능**:
  - Stable Diffusion 3 파이프라인
  - 메모리 효율적 생성
  - 이미지 품질 검증
  - 배치 생성 지원

#### CLIP Reward Calculator (`models/clip_reward.py`)

- **목적**: 텍스트-이미지 유사도 기반 보상 계산
- **기능**:
  - 단일/배치 보상 계산
  - 다중 보상 함수 (유사도, 품질, 일관성)
  - 보상 정규화 및 스케일링

### 2. Training 폴더 - GRPO 학습 알고리즘

#### GRPO Trainer (`training/grpo_trainer.py`)

- **목적**: Group Relative Policy Optimization 구현
- **GRPO vs PPO 차이점**:
  - PPO: 개별 샘플 기반 어드밴티지
  - GRPO: 그룹 내 상대적 어드밴티지 (더 안정적)
- **핵심 기능**:
  - 그룹 기반 어드밴티지 계산
  - 참조 모델 관리
  - KL 발산 페널티
  - 클리핑된 서로게이트 손실

### 3. Utils 폴더 - 유틸리티

#### Data Loader (`utils/data_loader.py`)

- **목적**: 학습/검증 데이터 관리
- **기능**:
  - JSONL 형식 데이터 로딩
  - 카테고리/난이도별 배치 생성
  - 균형잡힌 배치 생성
  - 결과 저장

### 4. Integration 폴더 - 통합 시스템

#### Main Trainer (`integration/main_trainer.py`)

- **목적**: 전체 시스템 통합 및 실행
- **기능**:
  - 모든 컴포넌트 초기화
  - End-to-End 학습 파이프라인
  - 실시간 모니터링
  - 체크포인트 관리

## 사용법 🛠️

### 1. 기본 설정

```python
# 기본 설정으로 시스템 실행
from integration.main_trainer import VLMGRPOSystem

system = VLMGRPOSystem()
system.initialize_components()
system.run_training()
```

### 2. 커스텀 설정

```python
# 커스텀 설정 파일 사용
system = VLMGRPOSystem("config/my_config.json")
```

### 3. 개별 컴포넌트 사용

```python
# VLM만 사용
from models.vlm_wrapper import VLMWrapper

vlm = VLMWrapper()
enhanced = vlm.enhance_prompt("a cat")
print(enhanced)

# SD3만 사용
from models.sd_generator import SD3Generator

generator = SD3Generator()
image = generator.generate_image("beautiful landscape")

# CLIP 보상만 사용
from models.clip_reward import CLIPRewardCalculator

calculator = CLIPRewardCalculator()
reward = calculator.calculate_reward(image, "beautiful landscape")
```

## 설정 파일 ⚙️

`config/default_config.json`에서 모든 설정을 관리할 수 있습니다:

```json
{
  "model_settings": {
    "vlm_model": "microsoft/DialoGPT-medium",
    "sd_model": "runwayml/stable-diffusion-v1-5",
    "clip_model": "openai/clip-vit-base-patch32"
  },
  "training_settings": {
    "learning_rate": 1e-5,
    "group_size": 4,
    "num_iterations": 50
  }
}
```

## 데이터 형식 📊

### 학습 데이터 (train_prompts.jsonl)

```json
{"user_prompt": "a cat", "category": "basic", "difficulty": "easy"}
{"user_prompt": "sunset", "category": "basic", "difficulty": "easy"}
{"user_prompt": "abstract art", "category": "creative", "difficulty": "hard"}
```

### 검증 데이터 (val_prompts.jsonl)

```json
{"user_prompt": "dog", "category": "basic", "difficulty": "easy"}
{"user_prompt": "city skyline", "category": "photography", "difficulty": "medium"}
```

## 실험 추적 📈

Wandb를 통한 실시간 실험 추적:

- 학습 메트릭 (loss, reward, KL divergence)
- 검증 결과 (성공률, 품질 점수)
- 생성된 이미지 샘플
- 하이퍼파라미터 추적

## 출력 결과 📁

학습 완료 후 `vlm_grpo_results/` 폴더에 생성되는 파일들:

```
vlm_grpo_results/
├── best_model.pt              # 최고 성능 모델
├── checkpoint_iter_10.pt      # 주기적 체크포인트
├── final_results.json         # 최종 결과 요약
├── validation_iter_5.json     # 검증 결과
└── vlm_grpo_training.log      # 학습 로그
```

## 성능 최적화 🚀

### Apple Silicon (MPS) 지원

- 모든 모델이 Apple Silicon MPS를 자동 감지하고 활용
- GPU 메모리 효율적 사용

### 메모리 최적화

- Attention slicing
- Gradient checkpointing
- Mixed precision training

### 배치 처리

- 효율적인 배치 생성
- 병렬 처리 지원

## 주요 알고리즘 🧠

### GRPO (Group Relative Policy Optimization)

1. **그룹 데이터 수집**:

   ```
   프롬프트 배치 → VLM 개선 → SD3 생성 → CLIP 보상
   ```

2. **어드밴티지 계산**:

   ```python
   group_mean = np.mean(rewards)
   advantages = rewards - group_mean  # 상대적 성능
   ```

3. **정책 업데이트**:
   ```python
   ratio = π_θ / π_ref  # 정책 비율
   loss = -min(ratio * advantage, clipped_ratio * advantage)
   ```

## 모니터링 및 디버깅 🔍

### 로깅 레벨

- `DEBUG`: 상세한 디버깅 정보
- `INFO`: 일반적인 진행 상황
- `WARNING`: 경고 메시지
- `ERROR`: 오류 발생

### 주요 메트릭

- **Policy Loss**: 정책 손실
- **KL Divergence**: 참조 모델과의 차이
- **Average Reward**: 평균 보상
- **Success Rate**: 검증 성공률

## 문제 해결 🛠️

### 일반적인 문제들

1. **메모리 부족**:

   - 배치 크기 줄이기
   - `memory_efficient=True` 설정

2. **학습 불안정**:

   - Learning rate 줄이기
   - KL beta 조정

3. **낮은 보상**:
   - 보상 가중치 조정
   - 프롬프트 품질 확인

## 확장 가능성 🔮

### 새로운 모델 추가

- VLM: 다른 언어 모델로 교체 가능
- SD: 다른 diffusion 모델 지원
- CLIP: 다른 vision-language 모델 사용

### 새로운 보상 함수

- 미적 품질 평가
- 안전성 검사
- 스타일 일관성

### 다중 모달 확장

- 비디오 생성
- 3D 모델 생성
- 오디오 생성

## 라이센스 📄

이 프로젝트는 연구 및 교육 목적으로 제작되었습니다.

## 기여하기 🤝

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

---

**Happy Training! 🎉**

더 자세한 정보나 질문이 있으시면 이슈를 생성해 주세요.
