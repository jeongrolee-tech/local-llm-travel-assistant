"""선택 실습 A — Quantization 비교. 같은 모델의 양자화 버전 2개.

발제문 5절 「선택 실습 A. Quantization 비교」.
    - 같은 모델의 양자화 버전 2개를 질문·설정·실행 조건에 맞춰 비교한다
    - VRAM 사용량, 속도, 품질 변화를 실제 관측 결과로 설명한다
    - 서로 다른 모델의 차이를 양자화 효과로 해석하지 않는다

비교 대상은 gemma4 E4B-it 한 모델의 두 양자화본이다. `ollama show` 로
architecture(gemma4) · parameters(8.0B) · embedding length(2560) ·
context length(131072) 가 모두 같고 quantization 만 다름을 확인했다.

    gemma4:e4b          Q4_K_M   디스크  9.6 GB   digest c6eb396dbd59
    gemma4:e4b-it-q8_0  Q8_0     디스크 11   GB   digest 9dcc35808b42

두 버전을 **같은 세션에서 연속 실행**한다. 필수 실험의 Q4_K_M 기록을 그대로
가져다 쓰지 않는 이유는, VRAM 여유와 GPU 상태가 세션마다 달라 속도·적재 비교가
오염되기 때문이다. 대신 이번 Q4_K_M 재실행값을 필수 실험값과 나란히 두면
재현성 점검도 함께 된다.

설정·지시문·질문·기록 형식은 run_local.py 를 import 해서 쓴다. 같은 코드
경로를 타므로 "조건을 동일하게 맞췄다" 가 문서상의 주장이 아니라 코드로
보장된다.

    uv run python src/run_quant.py

출력
    results/quant_raw.jsonl      양자화 비교 회차 (필수 실험 40회와 분리)
    results/quant_warmup.jsonl   워밍업 (집계 제외)
    results/errors.jsonl         호출 실패 (필수 실험과 같은 파일, model 로 구분)
"""

import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ollama import Client

import run_local as base

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

RAW = os.path.join(RESULTS, "quant_raw.jsonl")
WARMUP = os.path.join(RESULTS, "quant_warmup.jsonl")
ERRORS = os.path.join(RESULTS, "errors.jsonl")

# 같은 베이스 모델의 두 양자화본. 순서는 4bit → 8bit.
PAIR = ["gemma4:e4b", "gemma4:e4b-it-q8_0"]


def already_done():
    if not os.path.exists(RAW):
        return 0
    return sum(1 for _ in io.open(RAW, encoding="utf-8"))


def run_one(client, model, sysp, questions, repeats):
    """run_local 과 같은 절차: 콜드 시작 → 워밍업 1회 → 본 회차."""
    print("=" * 66)
    print("모델 :", model)

    client.generate(model=model, keep_alive=0)          # 콜드 상태에서 시작
    try:
        el, r = base.call(client, model, sysp, base.WARMUP_PROMPT)
        rec = base.build_record(client, model, "WARMUP", "warmup", None, 0, True, el, r)
        rec["exercise"] = "quantization"
        base.append(WARMUP, rec)
        print("워밍업 : %.2f초 (로딩 %.2f초) | VRAM %s MiB | %s — 집계 제외"
              % (el, rec["load_s"], rec["vram_mib"], rec["processor"]))
    except Exception as e:
        base.append(ERRORS, base.error_record(model, "WARMUP", 0, True, 0.0, e))
        print("워밍업 실패 : %s — 기록 후 계속" % type(e).__name__)

    ok = 0
    attempts = 0
    for run in range(1, repeats + 1):
        for q in questions:
            attempts += 1
            try:
                el, r = base.call(client, model, sysp, q["question"])
            except Exception as e:
                base.append(ERRORS, base.error_record(model, q["id"], run, False, 0.0, e))
                print("  [%s run%d] 실패 %s" % (q["id"], run, type(e).__name__))
                continue
            rec = base.build_record(client, model, q["id"], q["type"], q.get("grounded"),
                                    run, False, el, r)
            rec["exercise"] = "quantization"     # 필수 실험 기록과 구분하는 표시
            base.append(RAW, rec)
            ok += 1
            speed = ("%.1f tok/s" % rec["gen_tok_per_s"]) if rec["gen_tok_per_s"] else "계산 불가"
            print("  [%s run%d] %.2fs | in %d + out %d | %s | %s"
                  % (q["id"], run, el, rec["prompt_tok"], rec["out_tok"],
                     speed, rec["done_reason"]))

    client.generate(model=model, keep_alive=0)          # 다음 버전을 위해 내린다
    print("성공 %d / 시도 %d" % (ok, attempts))
    print()
    return ok, attempts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--repeats", type=int, default=base.REPEATS)
    ap.add_argument("--force", action="store_true", help="기록이 있어도 이어서 추가")
    args = ap.parse_args()

    done = already_done()
    if done and not args.force:
        sys.exit("이미 양자화 비교 기록이 %d건 있습니다.\n"
                 "재실험이라면 --force 로 이어 붙이고 재실험분임을 별도로 표시하십시오.\n"
                 "처음부터 다시 하려면 %s 를 먼저 정리하십시오." % (done, RAW))

    sysp, questions = base.load_inputs()
    client = Client(host=args.host, timeout=600)        # Q8_0 은 CPU 분할 시 느려진다

    print("선택 실습 A — Quantization 비교")
    print("비교 대상 : %s" % " vs ".join(PAIR))
    print("질문      : %d개 × %d회 × %d버전 = %d회"
          % (len(questions), args.repeats, len(PAIR),
             len(questions) * args.repeats * len(PAIR)))
    print("생성 설정 : %s, think=%s  (필수 실험과 동일)" % (base.OPTIONS, base.THINK))
    print()

    for model in PAIR:
        run_one(client, model, sysp, questions, args.repeats)

    print("본 회차 :", RAW)
    print("워밍업  :", WARMUP)
    print()
    print("집계    : uv run python src/aggregate_quant.py")


if __name__ == "__main__":
    main()
