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

### 1차 시도 — 측정에 실패했습니다

| 회차 | 전체 응답 시간 |
|---|---|
| 1회차 | 9.56초 |
| 2회차 | 8.70초 |

차이가 0.86초뿐이었습니다. **직전에 다른 스크립트(`04_metrics.py`)를 실행해 모델이
이미 GPU에 올라가 있었기 때문**입니다. 1회차에도 로딩이 없었으므로 워밍업 효과가
측정에서 사라졌습니다.

당시 `03_measure_time.py`는 `load_duration`을 출력하지 않아 이 사실을 알 수 없었습니다.
스크립트에 로딩 시간 출력과 적재 상태 경고를 추가했습니다.

### 2차 시도 — `ollama stop` 후 재측정

```
$ ollama stop qwen3.5:9b
$ uv run python examples/03_measure_time.py    # 1회차 (콜드)
전체 응답 시간: 19.87초
  └ 모델 로딩 : 4.28초
  └ 나머지    : 15.59초  (프롬프트 처리 + 생성)
출력 토큰: 684

$ uv run python examples/03_measure_time.py    # 2회차 (웜)
전체 응답 시간: 7.78초
  └ 모델 로딩 : 0.00초
  └ 나머지    : 7.77초  (프롬프트 처리 + 생성)
출력 토큰: 359
```

| 회차 | 전체 응답 시간 | 모델 로딩 | 나머지 | 출력 토큰 | 생성 속도(환산) |
|---|---|---|---|---|---|
| 1회차 (콜드) | 19.87초 | **4.28초** | 15.59초 | 684 | 43.9 tok/s |
| 2회차 (웜) | 7.78초 | **0.00초** | 7.77초 | 359 | 46.2 tok/s |

**확인 결과 — 전체 차이 12.09초 중 로딩은 4.28초뿐입니다.**

나머지 7.81초는 **출력 길이 차이**에서 왔습니다. 684토큰 대 359토큰으로 거의 2배입니다.
생성 속도로 환산하면 43.9 vs 46.2 tok/s로 비슷합니다.

**여기서 배운 것 3가지**

1. **측정 전에 적재 상태를 확인해야 합니다.** `ollama ps`가 비어 있는지 보지 않으면
   워밍업 효과가 측정에서 통째로 사라집니다. 1차 시도가 그랬습니다.

2. **전체 응답 시간만 보고 회차를 비교할 수 없습니다.** `temperature=0.2`라 회차마다
   답변 길이가 달라지고, 길이가 다르면 시간도 달라집니다. 발제문 주의사항
   *"tokens/s만으로 품질이나 사용자 체감 속도를 판단하지 않습니다"* 의 반대 방향
   사례입니다 — 시간만 봐도 안 됩니다. **로딩 시간과 출력 토큰 수를 함께 기록해야**
   시간 차이를 해석할 수 있습니다.

3. **본 실험 스크립트는 회차별로 `load_duration`과 `eval_count`를 반드시 남겨야 합니다.**
   둘이 없으면 "이 회차가 왜 느렸는가"에 답할 수 없습니다.

---

## STEP 4. 측정값 꺼내기

```
$ ollama stop qwen3.5:9b
$ uv run python .\examples_metrics.py
전체 응답 시간 : 4.86 초
로딩 시간      : 4.29 초
입력 토큰      : 15
출력 토큰      : 25
생성 속도      : 56.9 tok/s
종료 사유      : stop
```

| 항목 | 값 |
|---|---|
| 전체 응답 시간 | 4.86초 |
| 로딩 시간 | 4.29초 |
| 입력 토큰 | 15 |
| 출력 토큰 | 25 |
| 생성 속도 | 56.9 tok/s |
| `done_reason` | `stop` |

**관찰 — 로딩이 전체의 88%입니다.**

전체 4.86초 중 로딩이 4.29초, 실제 생성은 **0.57초**입니다. 질문이 짧으면 측정값을
로딩이 지배합니다. 본 실험 40회에서는 모델이 상주하므로 이 비용을 첫 회만 부담하며,
**워밍업 1회를 집계에서 분리해야 하는 이유가 다시 확인됩니다.**

### `done_reason` — `stop` 과 `length` 의 차이

`num_predict`(출력 한도)만 바꿔 같은 질문을 실행했습니다.

```
=== num_predict 미설정 ===
  done_reason : stop
  출력 토큰   : 64
  답변        : 제공된 자료에 따르면 출발 29일 전부터 20일 전까지 취소할 경우 요금의
                10%가 수수료로 부과됩니다. 따라서 출발 25일 전에 취소하시는 것은 이
                기간에 해당하므로 요금 총액의 10%를 수수료로 ...

=== num_predict = 20 ===
  done_reason : length
  출력 토큰   : 20
  답변        : 제공된 자료에 따르면 출발 29일 전부터 20일 전까지 취소할
```

| `done_reason` | 의미 | 채점 시 처리 |
|---|---|---|
| `stop` | 모델이 스스로 답변을 마침 | 정상 채점 |
| `length` | **출력 한도에 걸려 문장 중간에 잘림** | 잘린 응답을 그대로 채점하면 안 됨 |

`length` 사례는 `"취소할"` 에서 끊겼습니다. 이런 응답에 "핵심 정보 누락 0점"을 주면
모델 능력이 아니라 **우리 설정** 을 채점하는 것이 됩니다.

**본 실험 반영:** 본 실험은 `num_predict`를 설정하지 않으므로(모델 기본값) `length`가
날 가능성은 낮지만, **회차마다 `done_reason`을 기록해** 잘린 응답을 식별할 수 있게
합니다. `length`가 나온 회차는 별도 표시하고 채점 근거에 그 사실을 남깁니다.

---

## STEP 5. VRAM과 실행 조건

`num_ctx` 만 바꿔 각각 `ollama stop` 후 실행했습니다.

| 항목 | `num_ctx=4096` | `num_ctx=8192` |
|---|---|---|
| VRAM (GPU 적재) | 5,236 MiB | 5,251 MiB |
| 전체 적재 | 5,236 MiB | **5,987 MiB** |
| 적재 상태 | **100% GPU** | **88% GPU / CPU 분할** |
| digest | `6488c96fa5fa` | `6488c96fa5fa` |
| quantization | Q4_K_M | Q4_K_M |
| 실제 context | 4096 | 8192 |
| 생성 속도 | 59.8 tok/s | **53.0 tok/s** |
| 로딩 시간 | 3.62초 | 4.60초 |

**확인 결과 — 8192에서 GPU 적재량은 거의 그대로이고 전체 적재만 늘었습니다.**

`size_vram` 은 5,236 → 5,251 MiB 로 15 MiB 밖에 안 늘었는데 `size` 는 5,236 →
5,987 MiB 로 **751 MiB** 늘었습니다. 늘어난 KV 캐시가 GPU 가 아니라 **CPU 로
갔다는 뜻**입니다. Ollama 가 8,151 MiB GPU 에 대해 GPU 할당을 5,250 MiB 근처에서
막고 초과분을 CPU 로 보냅니다.

**대가는 생성 속도입니다. 59.8 → 53.0 tok/s 로 11% 느려졌습니다.**

**본 실험 context 를 4096 으로 고정한 근거가 여기서 직접 확인됩니다.** 8192 를 쓰면
두 후보의 비교 조건이 "한쪽만 CPU 분할" 로 어긋나고, 속도 측정값도 GPU 성능이 아니라
분할 비율을 재게 됩니다.

이 값은 예비 실행 단계에서 측정한 값과 일치합니다
([pilot/vram_measurements.md](pilot/vram_measurements.md)). 같은 결과가 독립적으로
재현되었습니다.

> `size_vram` 은 **측정 시점의 값이며 최대 VRAM 이 아닙니다.**
> `ollama ps` 의 PROCESSOR 는 **CPU/GPU 적재 상태이며 GPU 이용률이 아닙니다.**
> 위 "88% GPU" 는 적재 비율이지 GPU 를 88% 쓰고 있다는 뜻이 아닙니다.

**주의:** 확인 후 `04_metrics.py` 의 `num_ctx` 를 **4096 으로 되돌렸습니다.**
8192 를 남겨두면 이후 단계가 본 실험 고정 설정과 다른 조건에서 측정됩니다.

---

## STEP 6. 지시문 + 정책 문서 + Q1

`01_ollama_chat.py` 의 `messages` 를 고정 지시문 + 정책 문서 + Q1 으로 교체했습니다.

```python
messages=[{"role": "system", "content": sysp},     # system_prompt.txt + policy.md
          {"role": "user",   "content": q["question"]}],
```

### 문서 투입 전후 비교 — 같은 모델, 같은 질문

| | STEP 2 (질문만) | STEP 6 (지시문 + 문서 + 질문) |
|---|---|---|
| 답변 | "여행사나 항공사에 따라 취소 수수료 정책은 상이…" | "제공된 자료에 따르면 출발 29일 전부터 20일 전까지 취소할 경우 **요금의 10%**가 수수료로 부과됩니다." |
| 제시한 수치 | `0% ~ 5%`, `1~2 만 원 내외`, `출발 14 일 전까지 무료` | `10%` (문서의 취소 수수료 규정 값) |
| 근거 | 없음 — 일반론 | **문서** |

**STEP 2에서 지어냈던 숫자가 전부 사라졌습니다.** 고정 지시문 규칙 1(*"문서에 근거가
없는 내용은 절대 지어내지 않는다"*)이 실제로 작동하는 것을 확인했습니다. P4가 막으려는
행동이 무엇이고 무엇이 그것을 억제하는지가 이 대비로 드러납니다.

### 토큰

| 항목 | STEP 2 (질문만) | STEP 6 (문서 포함) |
|---|---|---|
| 입력 토큰 | 25 | **979** |
| 출력 토큰 | 378 | **72** |
| 합계 / `num_ctx` | 403 / 4096 | **1,051 / 4096** |
| `done_reason` | `stop` | `stop` |

입력은 39배 늘고 **출력은 5분의 1로 줄었습니다.** 출력이 준 것은 지시문 규칙 2
(*"답변은 3문장 이내로 한다"*)가 걸렸기 때문입니다. STEP 1에서 지시문 없이 489토큰이
나왔던 것과 대비됩니다.

**P3 판정 — 통과.** 문서 + 질문 + 답변이 1,051 토큰으로 `num_ctx=4096` 의 26%만
사용했습니다. 가장 긴 질문(Q2, 입력 987)을 기준으로 해도 여유가 충분합니다.

### 그러나 정답은 아닙니다

| | 기대 | 실제 |
|---|---|---|
| 구간 식별 | 출발 29일 전~20일 전 | ✅ 맞음 |
| 비율 | 요금의 10% | ✅ 맞음 |
| **금액** | **89,000원** | ❌ **계산하지 않음** ("요금 총액의 10%") |
| 근거 항목명 | 취소 수수료 규정 | ❌ 밝히지 않음 |

루브릭 기준으로 보면 항목 2(핵심 정보 누락)와 항목 3(지시·형식 준수)에서 감점입니다.
규칙 3(*"금액이나 조건을 말할 때는 근거가 된 문서의 항목명을 함께 밝힌다"*)을 지키지
않았습니다.

**예비 실행에서 관찰한 `qwen3.5:9b` 의 Q1 판정(◐ — 구간은 맞으나 금액 미계산)이 그대로
재현되었습니다.** 로컬 A를 "금액을 계산하지만 과잉 회피하는 유형"으로 기술한 근거가
수동 실행에서도 확인됩니다.

> **주의:** `01_ollama_chat.py` 는 토큰 수를 출력하지 않습니다. 위 토큰 값은 같은
> 설정으로 별도 측정한 것입니다. 본 실험 스크립트는 회차마다 `prompt_eval_count` 와
> `eval_count` 를 반드시 기록해야 P3를 증명할 수 있습니다.

---

## STEP 7. 레코드 저장

실행 스크립트: [`examples/05_record.py`](../examples/05_record.py)
저장 경로: `results/practice/sample.jsonl`

```
$ ollama stop qwen3.5:9b
$ uv run python examples/05_record.py
저장 완료 → results\practice\sample.jsonl

  qid              Q1
  run              1
  warmup           False
  model            qwen3.5:9b
  digest           6488c96fa5fa
  quantization     Q4_K_M
  context_length   4096
  options          {'temperature': 0.2, 'num_ctx': 4096}
  think            False
  num_predict      None
  processor        100% GPU
  elapsed_s        5.417
  load_s           3.874
  prompt_tok       979
  out_tok          60
  gen_tok_per_s    58.84
  gen_speed_note   None
  vram_mib         5236
  status           success
  done_reason      stop
  answer           제공된 자료에 따르면 출발 29일 전부터 20일 전까지 취소할 경우 ...

채워지지 않은 필수 필드: 없음
```

### 되읽기 검증

저장한 JSONL 을 다시 파싱해 발제문 요구 항목이 모두 들어 있는지 대조했습니다.
레코드 21개 필드, 한글 보존 정상.

| 발제문 요구 | 필드 | |
|---|---|---|
| 질문·실행 ID | `qid` | OK |
| 반복 회차 | `run` | OK |
| 워밍업 여부 | `warmup` | OK |
| 모델 태그 | `model` | OK |
| digest | `digest` | OK |
| quantization_level | `quantization` | OK |
| 실제 context_length | `context_length` | OK |
| 생성 설정 | `options`, `think` | OK |
| 출력 한도 | `num_predict` | OK (미설정 = `None`) |
| CPU/GPU 적재 상태 | `processor` | OK |
| 전체 응답 시간 | `elapsed_s` | OK |
| 로딩 시간 | `load_s` | OK |
| 출력 토큰 수 | `out_tok` | OK |
| 생성 속도 | `gen_tok_per_s` | OK |
| VRAM | `vram_mib` | OK |
| 성공·오류 상태 | `status`, `done_reason` | OK |
| 원본 응답 | `answer` | OK |

**확인 결과: 채워지지 않은 필수 필드 없음.**

`None` 으로 남은 두 필드는 의도된 것입니다.

| 필드 | `None` 인 이유 |
|---|---|
| `num_predict` | 본 실험은 출력 한도를 설정하지 않습니다(모델 기본값). **설정하지 않았다는 사실 자체가 기록**이므로 필드를 지우지 않습니다. |
| `gen_speed_note` | `eval_duration > 0` 이라 생성 속도를 정상 계산했습니다. 계산 불가일 때만 사유가 들어갑니다. |

**측정 불가 값을 `0` 으로 채우지 않는 구조입니다.** 발제문 주의사항이 요구하는 바이며,
`gen_tok_per_s` 가 `None` 이면 `gen_speed_note` 에 사유가 남습니다. 집계 시 `None` 은
평균 계산에서 제외하고 `n` 에도 넣지 않습니다.

**이 레코드 구조가 그대로 `src/run_local.py` 의 출력 형식이 됩니다.**

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
| STEP 3 | 1·2회차 차이가 0.86초로 워밍업 효과가 안 보임 | 직전 스크립트 실행으로 모델이 이미 적재된 상태였음 | `ollama stop` 후 재측정. `03_measure_time.py`에 `load_duration` 출력과 적재 상태 경고 추가 |
| | | | |

---

## 본 실험 스크립트에 반영할 것

- [ ] 워밍업 1회를 `results/local_warmup.jsonl`로 분리
- [ ] 질문당 2회를 각각 저장 (좋은 쪽만 고르지 않음)
- [ ] `ResponseError`·`ReadTimeout`을 잡아 `results/errors.jsonl`로 분리
- [ ] `eval_duration <= 0`이면 생성 속도를 `None` + 사유로 기록
- [ ] 모델 교체 전 `keep_alive=0`으로 내리기
- [ ] 레코드에 digest·quantization·context_length·생성 설정·적재 상태 포함
