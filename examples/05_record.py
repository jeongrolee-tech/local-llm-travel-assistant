"""STEP 7 — 발제문이 요구하는 필드를 모두 담아 결과 1건을 파일로 저장한다.

이 레코드 구조가 그대로 src/run_local.py 의 출력 형식이 된다.
여기서 빠진 필드가 있으면 본 실험 40회를 다시 돌려야 한다.

실행:
    uv run python examples/05_record.py
"""

import io
import json
import os
from time import perf_counter

from ollama import Client

# ── 본 실험 고정 설정 (README 4절) ────────────────────────────────
MODEL = "qwen3.5:9b"
OPTIONS = {"temperature": 0.2, "num_ctx": 4096}   # num_predict 미설정 = 모델 기본값
THINK = False
OUT = os.path.join("results", "practice", "sample.jsonl")

client = Client(host="http://127.0.0.1:11434", timeout=180)

# ── 입력: 고정 지시문 + 정책 문서 + 질문 ──────────────────────────
policy = io.open("data/policy.md", encoding="utf-8").read()
sysp = io.open("data/system_prompt.txt", encoding="utf-8").read().replace("{policy}", policy)
q = json.load(io.open("data/questions.json", encoding="utf-8"))["questions"][0]

# ── 호출 ──────────────────────────────────────────────────────────
start = perf_counter()
r = client.chat(
    model=MODEL,
    messages=[{"role": "system", "content": sysp},
              {"role": "user", "content": q["question"]}],
    stream=False, think=THINK, options=OPTIONS,
)
elapsed = perf_counter() - start

# ── 생성 속도: 계산 불가면 0으로 채우지 않고 사유를 남긴다 ─────────
ok = bool(r.eval_duration and r.eval_duration > 0)
gen_speed = round(r.eval_count / (r.eval_duration / 1e9), 2) if ok else None
gen_note = None if ok else "eval_duration <= 0 — 계산 불가"

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
    "gen_tok_per_s": gen_speed,
    "gen_speed_note": gen_note,
    "vram_mib": None,
    # 결과
    "status": "success", "done_reason": r.done_reason,
    "answer": r.message.content,
}

# ── 적재 상태와 실행 조건은 ps() 에서 읽는다 ──────────────────────
for m in client.ps().models:
    if m.model == MODEL:
        rec["digest"] = m.digest[:12]
        rec["quantization"] = m.details.quantization_level
        rec["context_length"] = m.context_length
        rec["vram_mib"] = round(m.size_vram / 1048576)
        rec["processor"] = "100% GPU" if m.size_vram == m.size else "CPU 분할"

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with io.open(OUT, "a", encoding="utf-8") as f:
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ── 확인: 채워지지 않은 필드가 있는가 ─────────────────────────────
OPTIONAL = {"num_predict", "gen_speed_note"}   # 의도적으로 비어 있을 수 있는 필드
missing = [k for k, v in rec.items() if v is None and k not in OPTIONAL]

print("저장 완료 →", OUT)
print()
for k, v in rec.items():
    if k == "answer":
        v = (v or "").strip().replace("\n", " ")[:60] + " ..."
    print("  %-16s %s" % (k, v))
print()
print("채워지지 않은 필수 필드:", missing if missing else "없음")
