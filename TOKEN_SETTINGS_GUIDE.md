# 토큰 설정 통일 가이드

## 📋 개요

VLM GRPO 시스템에서 토큰 관련 설정을 통일하여 일관성과 유지보수성을 개선했습니다.

## 🔧 변경사항

### ✅ 통합된 토큰 설정

기존에 여러 곳에 분산되어 있던 토큰 설정을 `config/default_config.json`의 `token_settings` 섹션으로 통합했습니다.

```json
{
  "token_settings": {
    "_comment": "토큰 관련 통합 설정",
    "max_new_tokens": 25,
    "max_prompt_length": 77,
    "max_sequence_length": 102,
    "_description": "max_sequence_length = max_prompt_length + max_new_tokens"
  }
}
```

### 📊 각 설정의 의미

| 설정                  | 값  | 용도        | 설명                                  |
| --------------------- | --- | ----------- | ------------------------------------- |
| `max_new_tokens`      | 25  | GRPO 학습   | 정책 네트워크가 생성할 새로운 토큰 수 |
| `max_prompt_length`   | 77  | 입력 검증   | CLIP 모델의 77토큰 제한을 준수        |
| `max_sequence_length` | 102 | 메모리 할당 | 전체 시퀀스 길이 (77 + 25)            |

## 🗂️ 기존 설정 제거

### ❌ 제거된 중복 설정들

1. **`training_settings.vlm_policy_settings`** → `token_settings`로 통합
2. **`generation_settings.vlm_generation.max_new_tokens`** → `token_settings.max_new_tokens` 참조
3. **GRPO trainer의 하드코딩된 값들** → Config에서 동적 로드

### ✅ 간소화된 구조

```json
{
  "generation_settings": {
    "vlm_generation": {
      "_comment": "VLM 생성 파라미터 - token_settings에서 max_new_tokens 참조",
      "temperature": 0.8,
      "top_p": 0.9,
      "do_sample": true,
      "use_cache": true
    }
  }
}
```

## 🔄 코드 변경사항

### 1. **GRPO Trainer (`training/grpo_trainer.py`)**

```python
# 기존 (하드코딩)
max_new_tokens: int = 25
max_sequence_length: int = 100

# 변경 후 (Config 기반)
max_new_tokens: int = 25      # Config에서 로드됨
max_prompt_length: int = 77   # CLIP 제한
max_sequence_length: int = 102 # prompt + new_tokens
```

### 2. **Main Trainer (`integration/main_trainer.py`)**

```python
# 기존 (분산된 설정 참조)
max_new_tokens=self.config['generation_settings']['vlm_generation']['max_new_tokens']

# 변경 후 (통합된 설정 참조)
max_new_tokens=self.config['token_settings']['max_new_tokens']
```

### 3. **VLM Wrapper (`models/vlm_wrapper.py`)**

```python
# 기존 (하드코딩)
if max_new_tokens is None:
    max_new_tokens = 20

# 변경 후 (Config 기반)
if max_new_tokens is None:
    config = self._load_config()
    max_new_tokens = config.get('token_settings', {}).get('max_new_tokens', 20)
```

## 📈 장점

### 1. **일관성 보장**

- 모든 컴포넌트가 동일한 토큰 설정을 사용
- 설정 변경 시 한 곳에서만 수정하면 됨

### 2. **유지보수성 향상**

- 중복 설정 제거로 혼란 방지
- 명확한 설정 구조로 이해도 향상

### 3. **확장성 개선**

- 새로운 토큰 관련 설정 추가 시 일관된 위치
- 설정 검증 및 계산 로직 통합 가능

## 🎯 사용 방법

### Config에서 토큰 설정 접근

```python
# Python 코드에서
config = load_config()
max_new_tokens = config['token_settings']['max_new_tokens']
max_prompt_length = config['token_settings']['max_prompt_length']
max_sequence_length = config['token_settings']['max_sequence_length']
```

### 설정 변경

```json
{
  "token_settings": {
    "max_new_tokens": 30, // 더 긴 생성을 원할 때
    "max_prompt_length": 77, // CLIP 제한 (고정)
    "max_sequence_length": 107 // 77 + 30
  }
}
```

## ⚠️ 주의사항

### 1. **CLIP 제한**

- `max_prompt_length`는 77로 고정 (CLIP 모델 제한)
- 이 값을 변경하면 CLIP 보상 계산에 문제 발생 가능

### 2. **메모리 고려사항**

- `max_sequence_length` 증가 시 메모리 사용량 증가
- GPU 메모리 한계를 고려하여 설정

### 3. **학습 효율성**

- `max_new_tokens`가 너무 크면 학습이 불안정해질 수 있음
- 25-30 토큰이 적절한 범위

## 🔍 검증 방법

### 설정 일관성 확인

```python
# 자동 검증 스크립트
def validate_token_settings(config):
    token_settings = config['token_settings']

    max_new = token_settings['max_new_tokens']
    max_prompt = token_settings['max_prompt_length']
    max_seq = token_settings['max_sequence_length']

    assert max_seq == max_prompt + max_new, f"Sequence length mismatch: {max_prompt} + {max_new} ≠ {max_seq}"
    assert max_prompt == 77, f"CLIP constraint violated: {max_prompt} ≠ 77"

    print("✅ Token settings validation passed!")
```

## 🎉 결론

토큰 설정 통일을 통해:

- ✅ **일관성**: 모든 컴포넌트가 동일한 설정 사용
- ✅ **명확성**: 각 설정의 역할과 제약사항 명확화
- ✅ **유지보수성**: 중복 제거로 관리 편의성 향상
- ✅ **확장성**: 새로운 토큰 관련 기능 추가 시 일관된 구조

이제 토큰 관련 설정을 수정할 때 `config/default_config.json`의 `token_settings` 섹션만 수정하면 됩니다! 🚀
