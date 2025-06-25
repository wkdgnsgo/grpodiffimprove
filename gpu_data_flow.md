# GPU 데이터 이동 및 처리 흐름

## GPU 배치 전략

### GPU 0: QWEN VL 모델 및 정책 네트워크

- **QWEN VL 모델**: 토큰 임베딩, 텍스트 생성
- **정책 네트워크 (Policy Head)**: 액션 선택, 그라디언트 업데이트
- **토큰화**: 입력 텍스트를 토큰으로 변환
- **최종 그라디언트 업데이트**: 모든 계산 결과를 받아 모델 파라미터 업데이트

### GPU 1: Stable Diffusion 3

- **이미지 생성**: 강화된 프롬프트로부터 이미지 생성
- **메모리 집약적 작업**: SD3 파이프라인 실행

### GPU 2: CLIP 리워드 모델

- **리워드 계산**: 원본 프롬프트, 강화된 프롬프트, 생성된 이미지 간 유사도 계산
- **CLIP 인코딩**: 텍스트와 이미지 임베딩 생성

## 데이터 이동 흐름

### 1. 초기화 단계

```python
# GPU 0: QWEN VL 모델 및 정책 헤드
self.qwen_device = "cuda:0"
self.policy_device = "cuda:0"

# GPU 1: Stable Diffusion 3
self.sd_device = "cuda:1"

# GPU 2: CLIP 리워드
self.reward_device = "cuda:2"
```

### 2. 환경 리셋 (reset)

```python
# 입력 프롬프트 토큰화 → GPU 0으로 이동
tokens = self.tokenizer.encode(prompt, return_tensors="pt")
return {
    'input_ids': tokens.squeeze(0).to("cuda:0"),
    'attention_mask': attention_mask.squeeze(0).to("cuda:0")
}
```

### 3. 액션 선택 (get_action_and_log_prob)

```python
# 입력 텐서를 GPU 0으로 이동
input_ids = input_ids.to("cuda:0")
attention_mask = attention_mask.to("cuda:0")

# QWEN VL 모델에서 히든 스테이트 추출 (GPU 0)
outputs = self.qwen_model.model(input_ids, attention_mask)
hidden_states = outputs.last_hidden_state

# 정책 헤드에서 액션 선택 (GPU 0)
policy_logits = self.policy_head(last_hidden)
```

### 4. 환경 스텝 (step)

```python
# 에피소드 종료 시 리워드 계산
if done:
    # 이미지 생성 (GPU 1)
    with torch.cuda.device(1):
        result = self.sd_pipeline(prompt=enhanced_prompt)
        image = result.images[0]

    # 리워드 계산 (GPU 2)
    with torch.cuda.device(2):
        reward = self.reward_model.calculate_reward(
            original_prompt, enhanced_prompt, image
        )

    # 다음 상태를 GPU 0으로 이동
    next_state = {
        'input_ids': tokens.to("cuda:0"),
        'attention_mask': mask.to("cuda:0")
    }
```

### 5. 학습 스텝 (train_step)

```python
# 모든 배치 데이터를 GPU 0으로 이동
input_ids = torch.stack(padded_input_ids).to("cuda:0")
attention_masks = torch.stack(padded_attention_masks).to("cuda:0")
actions = torch.tensor(actions).to("cuda:0")
advantages = torch.tensor(advantages).to("cuda:0")

# 정책 로짓 계산 (GPU 0)
policy_logits = self.policy(input_ids, attention_masks)

# 손실 계산 및 역전파 (GPU 0)
total_loss.backward()
self.optimizer.step()
```

## 메모리 최적화

### GPU 메모리 분배

- **GPU 0**: QWEN VL (7B) + 정책 헤드 → 약 14GB
- **GPU 1**: Stable Diffusion 3 → 약 12GB
- **GPU 2**: CLIP 모델 → 약 2GB

### 메모리 절약 기법

1. **torch.cuda.device() 컨텍스트**: 임시 GPU 전환
2. **torch.no_grad()**: 그라디언트 비활성화
3. **주기적 메모리 정리**: `torch.cuda.empty_cache()`
4. **배치 크기 조정**: GPU 메모리에 맞게 조정

## 동기화 및 오류 처리

### 텐서 디바이스 일관성

```python
# 모든 텐서가 같은 디바이스에 있는지 확인
assert input_ids.device == attention_mask.device == torch.device("cuda:0")
```

### 오류 처리

```python
try:
    # GPU 작업 수행
    result = model(input_tensor.to(device))
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        torch.cuda.empty_cache()
        # 배치 크기 줄이기
    elif "Expected all tensors to be on the same device" in str(e):
        # 텐서 디바이스 재확인
        input_tensor = input_tensor.to(correct_device)
```

## 성능 최적화

### 병렬 처리

- **GPU 1과 2**: 이미지 생성과 리워드 계산을 병렬로 수행 가능
- **GPU 0**: 다음 액션 선택을 미리 준비

### 데이터 파이프라이닝

1. GPU 0에서 액션 선택
2. GPU 1에서 이미지 생성 시작
3. GPU 2에서 리워드 계산 준비
4. 결과를 GPU 0으로 수집하여 그라디언트 업데이트

## 주요 수정 사항

### 1. PureGRPOPolicy 클래스

- GPU 디바이스 설정 추가
- forward 메서드에서 입력 텐서 GPU 이동 처리
- 정책 헤드를 GPU 0에 명시적 배치

### 2. PureGRPOPromptEnvironment 클래스

- GPU 디바이스 설정 추가
- reset/step 메서드에서 적절한 GPU로 데이터 이동
- 이미지 생성과 리워드 계산 시 GPU 컨텍스트 사용

### 3. train_step 메서드

- 모든 배치 데이터를 GPU 0으로 이동
- 패딩 텐서 생성 시 올바른 디바이스 지정
- 그라디언트 계산과 업데이트를 GPU 0에서 수행

이러한 구조로 각 GPU가 전문화된 작업을 수행하면서도 데이터 일관성을 유지할 수 있습니다.
