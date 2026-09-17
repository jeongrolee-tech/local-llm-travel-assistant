# 수동 실행 워크스루 (발제문 STEP 4)

본 실험 전에 한 단계씩 손으로 돌려보며 기록 항목을 확인하는 순서입니다.
발제문 **STEP 4. 모델 실행 환경 확인과 Python 연결** 에 대응하며, 여기서 만든 조각이
그대로 `src/run_local.py`가 됩니다.

> **연습 결과는 `results/`에 쓰지 않습니다.** `results/practice/`에 씁니다.
> 집계 원칙이 "워밍업·재시도·추가 실험은 본 실험과 별도로 기록"이므로,
> 연습 산출물이 본 실험 폴더에 섞이면 안 됩니다.

> **`data/questions.json`은 수정하지 않습니다.** 고정 질문 10개는 첫 호출 이전에
> 확정·커밋(`7800589`)했습니다. 바꾸는 것은 `examples/01_ollama_chat.py`의
> `QUESTION` **변수**뿐입니다.

---

## STEP 0. 환경 확인

```bash
cd C:\Users\jeong\KANT\official-project1\local-llm-travel-assistant
uv sync
mkdir results\practice

ollama --version
ollama list
ollama ps
```

**확인:** `ollama ps`가 비어 있어야 합니다. 앞 실험이 남아 있으면 로딩 시간이 0으로
측정되어 워밍업 구분이 무의미해집니다.

> `ollama` 파이썬 패키지는 `__version__`을 노출하지 않습니다. 버전은 아래처럼
> `importlib.metadata.version()`으로 읽습니다.

**기록할 것 (발제문 STEP 4):** Python 버전, Ollama 버전, 주요 패키지 버전, GPU/VRAM.

```bash
uv run python -c "import sys; from importlib.metadata import version; print(sys.version); print('ollama', version('ollama')); print('openai', version('openai'))"
nvidia-smi --query-gpu=name,memory.total --format=csv
```

---

## STEP 1. CLI 대화 (발제문: "CLI 대화 성공과 Python 호출 성공을 각각 확인")

```bash
ollama run qwen3.5:9b --think=false "출발 25일 전에 취소하면 수수료가 얼마인가요?"
ollama ps
```

**확인:** `PROCESSOR`가 `100% GPU`인지, `CONTEXT`가 얼마인지.

`--think=false`를 뺀 채로 한 번 더 돌려보십시오. 훨씬 오래 걸립니다.
이것이 README 4절에서 thinking을 끈 이유입니다.

```bash
ollama stop qwen3.5:9b
```

---

## STEP 2. Python 호출 — `MODEL`과 `QUESTION`을 모두 바꾼다

[`examples/01_ollama_chat.py`](../examples/01_ollama_chat.py)에서 두 줄을 고칩니다.

```python
MODEL = "qwen3.5:9b"
QUESTION = "출발 25일 전에 취소하면 수수료가 얼마인가요?"   # data/questions.json 의 Q1
```

```bash
uv run python examples/01_ollama_chat.py
```

**확인:** 답변이 나오는가. 그리고 이 답변이 왜 이렇게 나왔는지 설명할 수 있는가.

> 정책 문서를 아직 안 넣었으므로 근거 없는 일반론이 나옵니다. 정상입니다.
> 문서를 넣는 것은 STEP 6입니다.

---

## STEP 3. 응답 시간 측정과 워밍업의 의미

```bash
uv run python examples/03_measure_time.py
uv run python examples/03_measure_time.py
```

**확인:** 1회차와 2회차의 시간 차이. 1회차에는 모델 로딩이 포함됩니다.

이것이 **워밍업 1회를 본 실험 집계에서 분리하는 이유**입니다.
발제문 집계 원칙: "첫 실행의 로딩 지연과 로드된 상태의 응답 지연을 구분합니다."

---

## STEP 4. 측정값 꺼내기 — `examples/04_metrics.py`

```python
from time import perf_counter
from ollama import Client

MODEL = "qwen3.5:9b"
OPTIONS = {"temperature": 0.2, "num_ctx": 4096}   # 출력 한도(num_predict)는 미설정
THINK = False

client = Client(host="http://127.0.0.1:11434", timeout=180)

start = perf_counter()
r = client.chat(
    model=MODEL,
    messages=[{"role": "user", "content": "안녕하세요"}],
    stream=False, think=THINK, options=OPTIONS,
)
elapsed = perf_counter() - start

print("전체 응답 시간 :", round(elapsed, 2), "초")
print("로딩 시간      :", round(r.load_duration / 1e9, 2), "초")
print("입력 토큰      :", r.prompt_eval_count)
print("출력 토큰      :", r.eval_count)
print("종료 사유      :", r.done_reason)

# 측정 불가 값을 0으로 채우지 않습니다 (발제문 명시)
if r.eval_duration and r.eval_duration > 0:
    print("생성 속도      :", round(r.eval_count / (r.eval_duration / 1e9), 1), "tok/s")
else:
    print("생성 속도      : 계산 불가 — eval_duration <= 0")
```

**확인:** `done_reason`. `stop`이면 정상 종료, `length`면 출력 한도에 걸려 잘린 것입니다.

**발제문 Performance 항목 대응**

| 발제문 요구 | 코드 |
|---|---|
| 전체 응답 시간 | `elapsed` |
| 모델 로딩 시간 | `r.load_duration / 1e9` |
| 토큰 생성 속도 | `r.eval_count / (r.eval_duration / 1e9)` |
| 통계 없거나 `eval_duration <= 0` | 계산하지 않고 사유 기록 |

---

## STEP 5. VRAM과 실행 조건 읽기

STEP 4 파일 아래에 붙입니다.

```python
for m in client.ps().models:
    if m.model == MODEL:
        vram = m.size_vram
        total = m.size
        print("VRAM (GPU 적재):", round(vram / 1048576), "MiB")
        print("전체 적재      :", round(total / 1048576), "MiB")
        print("적재 상태      :", "100% GPU" if vram == total
              else "%.0f%% GPU / CPU 분할" % (vram / total * 100))
        print("digest         :", m.digest[:12])
        print("quantization   :", m.details.quantization_level)
        print("실제 context   :", m.context_length)
```

**확인:** `size_vram == size` 이면 100% GPU입니다.

`OPTIONS`의 `num_ctx`를 `8192`로 바꿔 다시 돌려보십시오. `qwen3.5:9b`는 여기서
둘이 어긋납니다. 실험 context를 4096으로 고정한 근거입니다.

> `size_vram`은 **측정 시점의 값이며 최대 VRAM이 아닙니다.**
> `ollama ps`의 PROCESSOR는 **CPU/GPU 적재 상태이며 GPU 이용률이 아닙니다.**

---

## STEP 6. 진짜 입력 넣기 — 지시문 + 정책 문서 + Q1

```python
import io, json

policy = io.open("data/policy.md", encoding="utf-8").read()
sysp = io.open("data/system_prompt.txt", encoding="utf-8").read().replace("{policy}", policy)
qs = json.load(io.open("data/questions.json", encoding="utf-8"))["questions"]
q = qs[0]

start = perf_counter()
r = client.chat(
    model=MODEL,
    messages=[{"role": "system", "content": sysp},
              {"role": "user", "content": q["question"]}],
    stream=False, think=THINK, options=OPTIONS,
)
elapsed = perf_counter() - start

print("질문:", q["question"])
print("기대:", q["expected"])
print("답변:", r.message.content)
print("입력 토큰:", r.prompt_eval_count)
```

**확인:** `prompt_eval_count`가 10 남짓에서 **980 근처로 뜁니다.** 정책 문서가 들어간 것입니다.

이 값을 기록해야 **P3(문서 + 질문 + 답변이 실험 Context Length 내 수용)** 을 증명할 수 있습니다.

---

## STEP 7. 한 건을 파일로 저장 (발제문 STEP 4 필수 산출)

```python
rec = {
    # 식별
    "qid": q["id"], "run": 1, "warmup": False,
    "model": MODEL, "digest": None, "quantization": None,
    # 실행 조건 (발제문 STEP 6 "실행 조건")
    "context_length": None,
    "options": OPTIONS, "think": THINK, "num_predict": OPTIONS.get("num_predict"),
    "processor": None,
    # 측정값
    "elapsed_s": round(elapsed, 3),
    "load_s": round(r.load_duration / 1e9, 3),
    "prompt_tok": r.prompt_eval_count,
    "out_tok": r.eval_count,
    "gen_tok_per_s": (round(r.eval_count / (r.eval_duration / 1e9), 2)
                      if r.eval_duration and r.eval_duration > 0 else None),
    "gen_speed_note": None if (r.eval_duration and r.eval_duration > 0)
                      else "eval_duration <= 0 — 계산 불가",
    "vram_mib": None,
    # 결과
    "status": "success", "done_reason": r.done_reason,
    "answer": r.message.content,
}

for m in client.ps().models:
    if m.model == MODEL:
        rec["digest"] = m.digest[:12]
        rec["quantization"] = m.details.quantization_level
        rec["context_length"] = m.context_length
        rec["vram_mib"] = round(m.size_vram / 1048576)
        rec["processor"] = "100% GPU" if m.size_vram == m.size else "CPU 분할"

with io.open("results/practice/sample.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("저장 완료")
```

**이 레코드가 발제문 요구를 모두 담는지 대조하십시오.**

| 발제문 요구 | 필드 |
|---|---|
| 질문·실행 ID | `qid`, `run` |
| 반복 회차 | `run` |
| 워밍업 여부 | `warmup` |
| 모델 태그 | `model` |
| **digest** | `digest` |
| **quantization_level** | `quantization` |
| **실제 context_length** | `context_length` |
| **출력 한도·생성 설정** | `options`, `think`, `num_predict` |
| **CPU/GPU 적재 상태** | `processor` |
| 전체 응답 시간 | `elapsed_s` |
| 로딩 시간 | `load_s` |
| 출력 토큰 수 | `out_tok` |
| 생성 속도 | `gen_tok_per_s` (불가 시 `None` + 사유) |
| VRAM | `vram_mib` |
| 성공·오류 상태 | `status`, `done_reason` |
| 원본 응답 | `answer` |

---

## STEP 8. 오류를 일부러 내고 기록한다

발제문 STEP 4: "**오류 증상과 확인 내용**을 기록합니다."

```python
import httpx

CASES = ["qwen3.5:999b", "존재하지-않는-모델:99b"]
for name in CASES:
    try:
        client.chat(model=name,
                    messages=[{"role": "user", "content": "안녕"}], stream=False)
    except Exception as e:
        print("%-24s -> %s: %s" % (name, type(e).__name__, e))

# 타임아웃도 확인합니다 (본 실험에서 실제로 날 수 있는 실패입니다)
slow = Client(host="http://127.0.0.1:11434", timeout=0.001)
try:
    slow.chat(model=MODEL, messages=[{"role": "user", "content": "안녕"}], stream=False)
except Exception as e:
    print("%-24s -> %s: %s" % ("timeout", type(e).__name__, e))
```

**실제 출력 (검증 완료)**

```
qwen3.5:999b             -> ResponseError: model 'qwen3.5:999b' not found (status code: 404)
존재하지-않는-모델:99b     -> ResponseError: invalid model name (status code: 400)
timeout                  -> ReadTimeout: timed out
```

**확인:** 오류 종류가 셋으로 갈립니다. 존재하지 않는 태그(404), 형식이 잘못된
이름(400), 타임아웃(`ReadTimeout`). 본 실험 스크립트는 `ollama.ResponseError`와
`httpx.ReadTimeout`을 모두 잡아야 40회가 중간에 멈추지 않습니다. 본 실험 스크립트는 이 예외를 잡아
**품질 점수와 분리해** `errors.jsonl`에 남겨야 합니다.

> 발제문 Quality 원칙: "호출 실패는 품질 점수와 별도로 기록하고
> **성공 응답으로 대체하지 않습니다.**"

---

## STEP 9. 모델을 바꿔 같은 것을 한다

발제문: "기본 실험은 **모델을 한 번에 하나씩** 실행하고 같은 PC에서 두 후보를 비교합니다."

```python
client.generate(model="qwen3.5:9b", keep_alive=0)   # 반드시 내리고 바꾼다
MODEL = "gemma4:e4b"
```

STEP 6~7을 그대로 다시 돌립니다.

**확인:** 같은 입력인데 `prompt_eval_count`가 980 → **1,074**로 늘어납니다.
토크나이저가 다르기 때문입니다.

---

## 다 하고 나면

이 9단계가 `src/run_local.py`가 할 일의 전부입니다. 남는 것은 반복뿐입니다.

```
모델 2개 × { 워밍업 1회(집계 분리) + 질문 10개 × 2회 }
  = 워밍업 2회 + 본 실험 40회
```

추가로 필요한 것은 세 가지입니다.

1. 워밍업 레코드를 `results/local_warmup.jsonl`로 분리
2. 실패 호출을 `results/errors.jsonl`로 분리 (성공 응답으로 대체 금지)
3. 같은 질문의 2회 응답을 **각각** 저장 (좋은 쪽만 고르지 않음)

연습이 끝나면 `results/practice/`는 지우거나, 남긴다면 본 실험이 아님을 명시하십시오.
