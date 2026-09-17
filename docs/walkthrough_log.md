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
ollama version is 0.34.0

$ ollama list
NAME                                                   ID              SIZE      MODIFIED
kanana2-3b-chatml:q4km                                 ab937df3c73d    2.2 GB    15 hours ago
hf.co/mradermacher/kanana-2-3b-instruct-GGUF:Q4_K_M    d5a16a4a92bc    2.2 GB    16 hours ago
exaone3.5:7.8b                                         c7c4e3d1ca22    4.8 GB    16 hours ago
gemma4:e2b                                             7fbdbf8f5e45    7.2 GB    18 hours ago
qwen3.5:4b                                             2a654d98e6fb    3.4 GB    18 hours ago
gemma4:e4b                                             c6eb396dbd59    9.6 GB    18 hours ago
qwen3.5:9b                                             6488c96fa5fa    6.6 GB    19 hours ago
gemma3:4b                                              a2af6cc3eb7f    3.3 GB    8 days ago
qwen3:4b-instruct-2507-q4_K_M                          0edcdef34593    2.5 GB    8 days ago

$ ollama ps
NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL
(비어 있음)
```

> `gemma3:4b`, `qwen3:4b-instruct-2507-q4_K_M` 은 선행 가이드 연습용이고
> `kanana2-3b-chatml:q4km` 은 오판으로 재빌드한 것입니다. 셋 다 본 실험 후보가 아닙니다.
> 본 실험 대상은 `qwen3.5:9b`(로컬 A)와 `gemma4:e4b`(로컬 B)입니다.

### 정리

| 항목 | 값 |
|---|---|
| OS | Windows 11 Pro 10.0.26200 |
| Python | 3.12.13 |
| 패키지 관리 | uv |
| `ollama` (Python 패키지) | 0.6.2 |
| `openai` (Python 패키지) | 3.8.0 |
| Ollama (서버) | 0.34.0 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| VRAM | 8,151 MiB |
| 시스템 RAM | 31.4 GB |

**확인 결과:** `ollama ps` 가 비어 있어 적재된 모델이 없는 상태에서 시작했습니다. `results/practice/` 디렉터리를 생성했습니다.

---

## STEP 1. CLI 대화

```
$ ollama ps
NAME          ID              SIZE      PROCESSOR    CONTEXT    UNTIL
qwen3.5:9b    6488c96fa5fa    5.5 GB    100% GPU     4096       4 minutes from now
```

### thinking 켬/끔 비교 (`--verbose`)

같은 질문을 `--think=false` 유무만 바꿔 실행했습니다. 각 실행 전 `ollama stop`으로 내렸습니다.

| 항목 | `--think=false` | thinking 켬 (기본값) | 차이 |
|---|---|---|---|
| total duration | **14.80s** | **45.78s** | **3.1배** |
| load duration | 3.89s | 4.60s | — |
| prompt eval count | 25 tok | 23 tok | — |
| eval count (출력) | 489 tok | **2,170 tok** | **4.4배** |
| eval duration | 10.79s | 41.04s | 3.8배 |
| eval rate | 45.32 tok/s | 52.87 tok/s | 생성 속도 자체는 비슷 |

**확인 결과**

| 항목 | 값 |
|---|---|
| PROCESSOR | `100% GPU` |
| CONTEXT | `4096` |
| SIZE | 5.5 GB |
| thinking 켰을 때 | 전체 시간 3.1배, 출력 토큰 4.4배 |

**관찰 3가지**

1. **생성 속도(tok/s)는 오히려 thinking 쪽이 약간 빠릅니다.** 느려진 이유는 속도가 아니라
   **토큰을 4.4배 더 뽑기 때문**입니다. 발제문 주의사항 *"tokens/s만으로 품질이나 체감
   속도를 판단하지 않습니다"* 가 그대로 확인됩니다.

2. **thinking 내용이 영어로 생성됩니다.** 한국어 질문인데 사고 과정은 전부 영어였습니다
   (`Okay, the user is asking about cancellation fees...`). 본 실험은 `think=False`로
   고정하므로 영향은 없지만, thinking을 켜는 구성을 검토한다면 고려할 점입니다.

3. **답변이 규정을 지어내지 않고 되물었습니다.** 정책 문서를 주지 않았으므로 항공사·예약
   경로를 확인해 달라고 답했습니다. 정상이며, 이 시점의 답변 품질은 평가 대상이 아닙니다.
   다만 **고정 지시문이 없어 "3문장 이내" 제약이 적용되지 않아** 489토큰이 나왔습니다.
   지시문과 정책 문서를 넣는 것은 STEP 6입니다.

---

## STEP 2. Python 호출

`examples/01_ollama_chat.py` 에서 변경한 줄:

```python
MODEL = "qwen3.5:9b"
QUESTION = "출발 25일 전에 취소하면 수수료가 얼마인가요?"   # data/questions.json 의 Q1
```

실행 설정:

```python
think=False
options={"temperature": 0, "num_ctx": 4096}   # num_predict 는 주석 처리 (미설정)
```

```
$ uv run python .\examples_ollama_chat.py
Ollama에 질문을 보냈습니다. 답변을 기다려 주세요.

[Ollama 답변]
여행사나 항공사에 따라 **취소 수수료 정책은 상이**하기 때문에, 정확한 금액을 알려드리기
위해서는 다음 정보를 확인해 주셔야 합니다.
1. 예약한 여행사 또는 항공사의 이름
2. 구매 날짜와 예약 유형
3. 여행 시작일과 현재 날짜의 차이
...
**일반적인 기준 (예시)**:
*   대부분의 국내/국제 항공사: 출발 2~3 주 전에 취소할 경우 항공권 가격의 0% ~ 5% 정도
*   패키지 여행: 출발 25 일 전이라면 보통 무료 취소이거나 1~2 만 원 내외
```

### CLI(STEP 1)와의 비교

| 항목 | CLI (STEP 1) | Python (STEP 2) |
|---|---|---|
| 모델 | `qwen3.5:9b` | `qwen3.5:9b` |
| thinking | `--think=false` | `think=False` |
| temperature | 모델 기본값 (1) | **0** |
| num_ctx | 4096 | 4096 |
| 답변 성격 | 항공사·예약 경로를 되물음 | 동일하게 되물음 |

**확인 결과: Python 호출 성공.** 발제문 체크리스트 7번 "CLI와 Python에서 모두 실행" 충족.

**관찰 — 문서가 없으면 그럴듯한 숫자를 만들어 냅니다.**

두 경로 모두 "정확한 금액은 확인이 필요하다"고 전제하면서도, **예시라는 이름으로 구체적인
수치를 제시했습니다** — `0% ~ 5%`, `1~2 만 원 내외`, `출발 14 일 전까지 무료 취소`.
우리 정책 문서에는 없는 숫자입니다.

이것이 **P4(문서에 없는 규정을 지어내지 않음)** 가 막으려는 행동이며, 고정 지시문
[`data/system_prompt.txt`](../data/system_prompt.txt)의 규칙 1(*"문서에 근거가 없는
내용은 절대 지어내지 않는다"*)이 이를 억제하는 장치입니다. STEP 6에서 지시문과 정책
문서를 넣은 뒤 같은 질문의 답이 어떻게 달라지는지 비교하십시오.

> **주의:** `01_ollama_chat.py` 의 `temperature: 0` 은 이 연습 파일의 값이며,
> 본 실험 고정 설정은 **`temperature: 0.2`** 입니다 (README 4절).
> `think=False` 옆의 `# 3문장 제한` 주석은 정확하지 않습니다. `think` 는 thinking 모드
> 스위치이고, "3문장 이내"는 고정 지시문 규칙 2에서 옵니다.

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
