"""로컬 본 실험 — 모델 1개 × 질문 10개 × 2회 = 20회.

발제문 STEP 6. 두 모델을 각각 실행해 합계 40회를 채운다.
모델은 한 번에 하나씩 실행한다.

    uv run python src/run_local.py --model qwen3.5:9b
    uv run python src/run_local.py --model gemma4:e4b

출력
    results/local_raw.jsonl      본 실험 회차 (집계 대상)
    results/local_warmup.jsonl   워밍업 1회 (집계 제외)
    results/errors.jsonl         호출 실패 (품질 점수와 분리)

설계 근거는 docs/walkthrough_log.md 에 단계별 실측과 함께 기록되어 있다.
"""

import argparse
import io
import json
import os
import sys
from datetime import datetime, timezone
from time import perf_counter

from ollama import Client

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

RAW = os.path.join(RESULTS, "local_raw.jsonl")
WARMUP = os.path.join(RESULTS, "local_warmup.jsonl")
ERRORS = os.path.join(RESULTS, "errors.jsonl")

# ── 본 실험 고정 설정 (README 4절). 모든 모델·모든 회차 동일 ──────
OPTIONS = {"temperature": 0.2, "num_ctx": 4096}   # num_predict 미설정 = 모델 기본값
THINK = False                                      # 전 모델 동일 (Qwen3.5 는 기본 켜짐)
REPEATS = 2                                        # 질문당 반복 회차
WARMUP_PROMPT = "안녕하세요"


def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_inputs():
    """고정 지시문 + 정책 문서 + 질문 10개. 어느 것도 수정하지 않는다."""
    policy = io.open(os.path.join(ROOT, "data", "policy.md"), encoding="utf-8").read()
    sysp = io.open(os.path.join(ROOT, "data", "system_prompt.txt"), encoding="utf-8").read()
    sysp = sysp.replace("{policy}", policy)
    qs = json.load(io.open(os.path.join(ROOT, "data", "questions.json"), encoding="utf-8"))
    return sysp, qs["questions"]


def runtime_info(client, model):
    """실행 조건을 ps() 에서 읽는다. 적재 상태는 측정 시점의 값이다."""
    info = {"digest": None, "quantization": None, "context_length": None,
            "vram_mib": None, "processor": None}
    try:
        for m in client.ps().models:
            if m.model == model:
                info["digest"] = m.digest[:12]
                info["quantization"] = m.details.quantization_level
                info["context_length"] = m.context_length
                info["vram_mib"] = round(m.size_vram / 1048576)
                info["processor"] = ("100% GPU" if m.size_vram == m.size
                                     else "CPU 분할 (%.0f%% GPU)" % (m.size_vram / m.size * 100))
    except Exception as e:                      # ps() 실패가 실험을 멈추게 하지 않는다
        info["processor"] = "확인 불가 (%s)" % type(e).__name__
    return info


def call(client, model, sysp, question):
    start = perf_counter()
    r = client.chat(
        model=model,
        messages=[{"role": "system", "content": sysp},
                  {"role": "user", "content": question}],
        stream=False, think=THINK, options=OPTIONS,
    )
    return perf_counter() - start, r


def build_record(client, model, qid, qtype, grounded, run, warmup, elapsed, r):
    # 측정 불가 값을 0 으로 채우지 않는다. 사유를 남긴다.
    ok = bool(r.eval_duration and r.eval_duration > 0)
    rec = {
        "ts": now(),
        "qid": qid, "qtype": qtype, "grounded": grounded,
        "run": run, "warmup": warmup,
        "model": model,
        "options": OPTIONS, "think": THINK, "num_predict": OPTIONS.get("num_predict"),
        "elapsed_s": round(elapsed, 3),
        "load_s": round(r.load_duration / 1e9, 3),
        "prompt_tok": r.prompt_eval_count,
        "out_tok": r.eval_count,
        "gen_tok_per_s": round(r.eval_count / (r.eval_duration / 1e9), 2) if ok else None,
        "gen_speed_note": None if ok else "eval_duration <= 0 — 계산 불가",
        "status": "success",
        "done_reason": r.done_reason,
        "answer": r.message.content,
    }
    rec.update(runtime_info(client, model))
    return rec


def error_record(model, qid, run, warmup, elapsed, exc):
    return {
        "ts": now(),
        "qid": qid, "run": run, "warmup": warmup, "model": model,
        "options": OPTIONS, "think": THINK,
        "elapsed_s": round(elapsed, 3),
        "status": "error",
        # 예외를 좁게 나열하면 예상 못 한 하나에 실험이 멈춘다.
        # 넓게 잡고 유형을 남겨 나중에 분류한다. (docs/walkthrough_log.md STEP 8)
        "error_type": type(exc).__name__,
        "error_msg": str(exc)[:500],
    }


def append(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def already_done(path, model):
    if not os.path.exists(path):
        return 0
    n = 0
    for line in io.open(path, encoding="utf-8"):
        try:
            if json.loads(line).get("model") == model:
                n += 1
        except ValueError:
            pass
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="실행할 모델 태그 (전체 태그)")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--force", action="store_true",
                    help="이 모델의 기록이 이미 있어도 이어서 추가")
    ap.add_argument("--out-dir", default=RESULTS,
                    help="출력 디렉터리. 스모크 테스트용으로만 바꾼다")
    args = ap.parse_args()

    global RAW, WARMUP, ERRORS
    if args.out_dir != RESULTS:
        RAW = os.path.join(args.out_dir, "local_raw.jsonl")
        WARMUP = os.path.join(args.out_dir, "local_warmup.jsonl")
        ERRORS = os.path.join(args.out_dir, "errors.jsonl")
        print("[스모크 테스트] 출력 → %s" % args.out_dir)
        print()

    model = args.model
    done = already_done(RAW, model)
    if done and not args.force:
        sys.exit("이미 %s 의 본 실험 기록이 %d건 있습니다.\n"
                 "재실험이라면 --force 로 이어 붙이고, 재실험분임을 별도로 표시하십시오.\n"
                 "처음부터 다시 하려면 %s 를 먼저 정리하십시오." % (model, done, RAW))

    sysp, questions = load_inputs()
    client = Client(host=args.host, timeout=300)

    print("모델        :", model)
    print("질문        : %d개 × %d회 = %d회" % (len(questions), args.repeats,
                                                len(questions) * args.repeats))
    print("생성 설정   : %s, think=%s" % (OPTIONS, THINK))
    print()

    # ── 워밍업 1회 — 집계에서 분리한다 ────────────────────────────
    client.generate(model=model, keep_alive=0)      # 콜드 상태에서 시작
    try:
        el, r = call(client, model, sysp, WARMUP_PROMPT)
        rec = build_record(client, model, "WARMUP", "warmup", None, 0, True, el, r)
        append(WARMUP, rec)
        print("워밍업 1회  : %.2f초 (로딩 %.2f초) — 집계 제외" % (el, rec["load_s"]))
    except Exception as e:
        append(ERRORS, error_record(model, "WARMUP", 0, True, 0.0, e))
        print("워밍업 실패 : %s — 기록 후 계속" % type(e).__name__)
    print()

    # ── 본 실험 ──────────────────────────────────────────────────
    ok_count = 0
    attempts = 0
    for run in range(1, args.repeats + 1):
        for q in questions:
            attempts += 1
            try:
                el, r = call(client, model, sysp, q["question"])
            except Exception as e:                  # 40회가 중간에 멈추면 안 된다
                append(ERRORS, error_record(model, q["id"], run, False, 0.0, e))
                print("  [%s run%d] 실패 %s" % (q["id"], run, type(e).__name__))
                continue
            rec = build_record(client, model, q["id"], q["type"], q.get("grounded"),
                               run, False, el, r)
            append(RAW, rec)
            ok_count += 1
            speed = ("%.1f tok/s" % rec["gen_tok_per_s"]) if rec["gen_tok_per_s"] else "속도 계산 불가"
            print("  [%s run%d] %.2fs | in %d + out %d | %s | %s"
                  % (q["id"], run, el, rec["prompt_tok"], rec["out_tok"],
                     speed, rec["done_reason"]))

    client.generate(model=model, keep_alive=0)      # 다음 모델을 위해 내린다

    print()
    print("성공 %d / 전체 시도 %d" % (ok_count, attempts))
    print("본 실험  :", RAW)
    print("워밍업   :", WARMUP)
    if os.path.exists(ERRORS):
        print("오류     :", ERRORS)


if __name__ == "__main__":
    main()
