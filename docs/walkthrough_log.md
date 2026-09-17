# 수동 실행 기록

[walkthrough.md](walkthrough.md)를 단계별로 수행하며 실제 출력을 붙여넣는 파일입니다.
발제문 STEP 4의 *"Python/Ollama/주요 패키지 버전, GPU/VRAM, 실행 설정, 오류 증상과
확인 내용을 기록합니다"* 에 해당하는 증빙입니다.

수행자: 단독 수행
시작일: 2026-09-17

---

## STEP 0. 환경 확인

### 버전

```
$ uv run python -c "import sys; from importlib.metadata import version; print(sys.version); print('ollama', version('ollama')); print('openai', version('openai'))"
3.12.13 (main, Jul 23 2026, 14:44:57) [MSC v.1944 64 bit (AMD64)]
ollama 0.6.2
openai 3.8.0
```

### GPU

```
$ nvidia-smi --query-gpu=name,memory.total --format=csv
name, memory.total [MiB]
NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB
```

### Ollama

```
$ ollama --version
(붙여넣기)

$ ollama list
(붙여넣기)

$ ollama ps
(붙여넣기 — 비어 있어야 정상)
```

### 정리

| 항목 | 값 |
|---|---|
| OS | Windows 11 Pro 10.0.26200 |
| Python | 3.12.13 |
| 패키지 관리 | uv |
| `ollama` (Python 패키지) | 0.6.2 |
| `openai` (Python 패키지) | 3.8.0 |
| Ollama (서버) | `{붙여넣기}` |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| VRAM | 8,151 MiB |
| 시스템 RAM | 31.4 GB |

**확인 결과:** `{ollama ps 가 비어 있었는가}`

---

## STEP 1. CLI 대화

```
$ ollama run qwen3.5:9b --think=false "출발 25일 전에 취소하면 수수료가 얼마인가요?"
(붙여넣기)

$ ollama ps
(붙여넣기)
```

**확인 결과**

| 항목 | 값 |
|---|---|
| PROCESSOR | `{}` |
| CONTEXT | `{}` |
| `--think=false` 없이 실행했을 때 체감 차이 | `{}` |

---

## STEP 2. Python 호출

`examples/01_ollama_chat.py` 에서 변경한 줄:

```python
MODEL = "{}"
QUESTION = "{}"
```

```
$ uv run python examples/01_ollama_chat.py
(붙여넣기)
```

**확인 결과:** `{정책 문서를 안 넣었을 때 답변이 어땠는가}`

---

## STEP 3. 응답 시간과 워밍업

```
$ uv run python examples/03_measure_time.py    # 1회차
(붙여넣기)

$ uv run python examples/03_measure_time.py    # 2회차
(붙여넣기)
```

| 회차 | 전체 응답 시간 |
|---|---|
| 1회차 (로딩 포함) | `{}` |
| 2회차 (로드된 상태) | `{}` |

**확인 결과:** `{차이가 얼마였고 왜 생겼는가}`

---

## STEP 4. 측정값 꺼내기

```
$ uv run python examples/04_metrics.py
(붙여넣기)
```

| 항목 | 값 |
|---|---|
| 전체 응답 시간 | `{}` |
| 로딩 시간 | `{}` |
| 입력 토큰 | `{}` |
| 출력 토큰 | `{}` |
| 생성 속도 | `{}` |
| `done_reason` | `{}` |

---

## STEP 5. VRAM과 실행 조건

| 항목 | `num_ctx=4096` | `num_ctx=8192` |
|---|---|---|
| VRAM (GPU 적재) | `{}` | `{}` |
| 전체 적재 | `{}` | `{}` |
| 적재 상태 | `{}` | `{}` |
| digest | `{}` | `{}` |
| quantization | `{}` | `{}` |
| 실제 context | `{}` | `{}` |

**확인 결과:** `{8192에서 무엇이 달라졌는가}`

---

## STEP 6. 지시문 + 정책 문서 + Q1

```
(붙여넣기 — 질문 / 기대 / 답변 / 입력 토큰)
```

| 항목 | 값 |
|---|---|
| 입력 토큰 (문서 없음, STEP 4) | `{}` |
| 입력 토큰 (문서 포함) | `{}` |
| 출력 토큰 | `{}` |
| 합계 / `num_ctx` | `{}` / 4096 |

**P3 판정:** `{문서 + 질문 + 답변이 4096 안에 들어갔는가}`

---

## STEP 7. 레코드 저장

저장 경로: `results/practice/sample.jsonl`

```json
(저장된 레코드 1건 붙여넣기)
```

**확인 결과:** `{None 으로 남은 필드가 있었는가}`

---

## STEP 8. 오류 확인

```
(붙여넣기)
```

| 오류 상황 | 예외 유형 | 메시지 |
|---|---|---|
| 존재하지 않는 태그 | `{}` | `{}` |
| 형식이 잘못된 이름 | `{}` | `{}` |
| 타임아웃 | `{}` | `{}` |

**본 실험 반영:** `{run_local.py 가 어떤 예외를 잡아야 하는가}`

---

## STEP 9. 모델 교체

```
(붙여넣기 — gemma4:e4b 로 STEP 6~7 재실행)
```

| 항목 | `qwen3.5:9b` | `gemma4:e4b` |
|---|---|---|
| 입력 토큰 (동일 입력) | `{}` | `{}` |
| 출력 토큰 | `{}` | `{}` |
| 전체 응답 시간 | `{}` | `{}` |
| VRAM | `{}` | `{}` |

**확인 결과:** `{입력 토큰이 왜 달랐는가}`

---

## 수행 중 만난 문제와 조치

| 단계 | 증상 | 원인 | 조치 |
|---|---|---|---|
| STEP 0 | `ollama.__version__` AttributeError | `ollama` 파이썬 패키지가 `__version__`을 노출하지 않음 | `importlib.metadata.version("ollama")`로 변경 |
| | | | |

---

## 본 실험 스크립트에 반영할 것

- [ ] 워밍업 1회를 `results/local_warmup.jsonl`로 분리
- [ ] 질문당 2회를 각각 저장 (좋은 쪽만 고르지 않음)
- [ ] `ResponseError`·`ReadTimeout`을 잡아 `results/errors.jsonl`로 분리
- [ ] `eval_duration <= 0`이면 생성 속도를 `None` + 사유로 기록
- [ ] 모델 교체 전 `keep_alive=0`으로 내리기
- [ ] 레코드에 digest·quantization·context_length·생성 설정·적재 상태 포함
