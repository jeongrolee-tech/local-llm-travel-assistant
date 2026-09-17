"""Cloud 비교 실험 — gpt-4o-mini × 공통 질문 5개 × 1회 = 5회.

발제문 STEP 7. 로컬은 질문당 2회, Cloud 는 1회이므로 반복 수가 다르다.
집계할 때 이 차이를 명시하고, 로컬의 좋은 결과 한 건만 골라 비교하지 않는다.

    uv run python src/run_cloud.py

API 키는 실행할 때 터미널에 입력한다. 화면에 표시되지 않고 파일에도 저장되지
않는다. 코드·저장소·로그·결과 파일 어디에도 키가 들어가지 않는다.

출력
    results/cloud_raw.jsonl   응답, 성공·오류 상태, 토큰, 응답 시간, 추정 비용
    results/errors.jsonl      호출 실패 (로컬과 같은 파일에 model 로 구분)
"""

import argparse
import io
import json
import os
import sys
from datetime import datetime, timezone
from getpass import getpass
from time import perf_counter

from openai import APIError, APITimeoutError, OpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
RAW = os.path.join(RESULTS, "cloud_raw.jsonl")
ERRORS = os.path.join(RESULTS, "errors.jsonl")

MODEL = "gpt-4o-mini"

# 로컬 본 실험과 같은 값 (README 4절). max_tokens 는 로컬의 num_predict 미설정에 대응해
# 넉넉히 두고, 잘림 여부는 finish_reason 으로 판별한다.
TEMPERATURE = 0.2
MAX_TOKENS = 1024

# 조사한 공시 단가 (docs/02_model_comparison.md 5절, 2026-09-16 기준).
# 여기서 계산하는 값은 "추정 비용" 이며 실제 청구 내역이 아니다.
# 실제 사용 내역은 https://platform.openai.com/usage 에서 확인한다.
PRICE_IN_PER_1M = 0.15
PRICE_OUT_PER_1M = 0.60


def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_inputs():
    """로컬 실험과 완전히 같은 지시문·문서·질문을 쓴다."""
    policy = io.open(os.path.join(ROOT, "data", "policy.md"), encoding="utf-8").read()
    sysp = io.open(os.path.join(ROOT, "data", "system_prompt.txt"), encoding="utf-8").read()
    sysp = sysp.replace("{policy}", policy)
    data = json.load(io.open(os.path.join(ROOT, "data", "questions.json"), encoding="utf-8"))
    subset = data["cloud_subset"]["question_ids"]
    by_id = {q["id"]: q for q in data["questions"]}
    return sysp, [by_id[qid] for qid in subset], data["cloud_subset"]


def append(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def already_done():
    if not os.path.exists(RAW):
        return 0
    return sum(1 for _ in io.open(RAW, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="기록이 이미 있어도 이어서 추가")
    args = ap.parse_args()

    done = already_done()
    if done and not args.force:
        sys.exit("이미 Cloud 실험 기록이 %d건 있습니다.\n"
                 "재실험이라면 --force 로 이어 붙이고 재실험분임을 별도로 표시하십시오.\n"
                 "처음부터 다시 하려면 %s 를 먼저 정리하십시오." % (done, RAW))

    sysp, questions, subset = load_inputs()

    print("모델        :", MODEL)
    print("질문        : %s (%d개 × 1회)" % (", ".join(q["id"] for q in questions), len(questions)))
    print("선정 규칙   :", subset["rule"])
    print("생성 설정   : temperature=%s, max_tokens=%s" % (TEMPERATURE, MAX_TOKENS))
    print()
    print("이 스크립트는 OpenAI API 를 %d회 호출합니다. 유료입니다." % len(questions))
    print()

    # 키는 여기서만 존재하고 어디에도 기록하지 않는다.
    api_key = getpass("OpenAI API 키를 붙여넣고 Enter (화면에 보이지 않음): ").strip()
    if not api_key:
        sys.exit("키를 입력하지 않아 API를 호출하지 않았습니다.")

    client = OpenAI(api_key=api_key, timeout=60, max_retries=0)
    print()

    ok = 0
    total_in = total_out = 0
    for q in questions:
        start = perf_counter()
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": sysp},
                          {"role": "user", "content": q["question"]}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
        except (APITimeoutError, APIError) as e:
            append(ERRORS, {
                "ts": now(), "model": MODEL, "qid": q["id"], "run": 1, "warmup": False,
                "status": "error", "error_type": type(e).__name__,
                "error_msg": str(e)[:500],   # 키는 메시지에 포함되지 않는다
            })
            print("  [%s] 실패 %s" % (q["id"], type(e).__name__))
            continue
        except Exception as e:
            append(ERRORS, {
                "ts": now(), "model": MODEL, "qid": q["id"], "run": 1, "warmup": False,
                "status": "error", "error_type": type(e).__name__, "error_msg": str(e)[:500],
            })
            print("  [%s] 실패 %s" % (q["id"], type(e).__name__))
            continue

        elapsed = perf_counter() - start
        u = r.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", None)
        cost = (u.prompt_tokens / 1e6 * PRICE_IN_PER_1M
                + u.completion_tokens / 1e6 * PRICE_OUT_PER_1M)
        ch = r.choices[0]

        append(RAW, {
            "ts": now(),
            "qid": q["id"], "qtype": q["type"], "grounded": q.get("grounded"),
            "run": 1, "warmup": False,
            "model": MODEL, "model_returned": r.model,
            "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
            "elapsed_s": round(elapsed, 3),          # 네트워크 왕복 포함
            "prompt_tok": u.prompt_tokens,
            "cached_prompt_tok": cached,
            "out_tok": u.completion_tokens,
            "total_tok": u.total_tokens,
            "est_cost_usd": round(cost, 6),          # 추정값. 실제 청구 내역이 아님
            "price_note": "공시 단가 in $%.2f / out $%.2f per 1M (2026-09-16 조사). "
                          "실제 사용 내역은 platform.openai.com/usage 에서 확인."
                          % (PRICE_IN_PER_1M, PRICE_OUT_PER_1M),
            "status": "success",
            "finish_reason": ch.finish_reason,
            "answer": ch.message.content,
        })
        ok += 1
        total_in += u.prompt_tokens
        total_out += u.completion_tokens
        print("  [%s] %.2fs | in %d + out %d | $%.6f | %s"
              % (q["id"], elapsed, u.prompt_tokens, u.completion_tokens, cost, ch.finish_reason))

    del api_key, client   # 키를 메모리에서 빨리 떨군다

    total_cost = total_in / 1e6 * PRICE_IN_PER_1M + total_out / 1e6 * PRICE_OUT_PER_1M
    print()
    print("성공 %d / 전체 시도 %d" % (ok, len(questions)))
    print("토큰 합계  : in %d + out %d" % (total_in, total_out))
    print("추정 비용  : $%.6f  (공시 단가 기준. 실제 청구 내역과 구분할 것)" % total_cost)
    print("실제 사용량: https://platform.openai.com/usage 에서 확인")
    print("결과       :", RAW)


if __name__ == "__main__":
    main()
