"""선택 실습 A — Quantization 비교. 같은 모델의 양자화 버전 2개.

이 스크립트가 하는 일을 한 줄로 쓰면 이렇다.

    같은 모델의 4bit 본과 8bit 본에 똑같은 질문 10개를 2회씩 던져서,
    VRAM·속도·품질이 어떻게 달라지는지 40회 분량을 기록한다.

발제문 5절 「선택 실습 A. Quantization 비교」가 요구하는 것.
    - 같은 모델의 양자화 버전 2개를 질문·설정·실행 조건에 맞춰 비교한다
    - VRAM 사용량, 속도, 품질 변화를 실제 관측 결과로 설명한다
    - 서로 다른 모델의 차이를 양자화 효과로 해석하지 않는다

── 양자화가 무엇이고 왜 비교하는가 ────────────────────────────────

모델은 숫자(가중치) 80억 개다. 그 숫자 하나를 몇 비트로 저장하느냐가
양자화다. 비트가 적을수록 표현할 수 있는 단계가 줄어 값이 뭉개진다.

    16bit  65,536단계   원본
     8bit     256단계   Q8_0
     4bit      16단계   Q4_K_M

뭉개는 대신 얻는 것은 용량과 속도다. 속도가 빨라지는 이유는 계산이 줄어서가
아니라, 토큰 하나를 만들 때마다 읽어야 하는 가중치 데이터가 절반이기
때문이다. 병목이 계산이 아니라 메모리 읽기 대역폭에 있다.

그래서 질문은 이것이다. **4bit는 싸고 빠른 대신 품질을 잃는가.**

── 비교 대상 ──────────────────────────────────────────────────────

비교 대상은 gemma4 E4B-it 한 모델의 두 양자화본이다.

    gemma4:e4b          Q4_K_M   디스크  9.6 GB   digest c6eb396dbd59
    gemma4:e4b-it-q8_0  Q8_0     디스크 11   GB   digest 9dcc35808b42

실행 전에 `ollama show` 로 architecture(gemma4) · parameters(8.0B) ·
embedding length(2560) · context length(131072) 가 모두 같고 quantization
만 다름을 확인했다. **이 확인이 비교의 전제다.** 이걸 건너뛰고 비교하면,
관측된 차이가 양자화 때문인지 애초에 다른 모델이라 그런지 구분할 수 없고,
발제문이 금지한 "서로 다른 모델의 차이를 양자화 효과로 해석" 이 된다.

bf16(16GB)과 qat 본은 제외했다. 이유는 docs/06_quantization.md 2절에 있다.

── 설계상 중요한 두 가지 ──────────────────────────────────────────

(1) 조건 동일성을 코드로 보장한다.

    지시문·질문·생성 설정·기록 형식을 run_local.py 에서 import 해서 쓴다.
    값을 복사해 두면 한쪽만 고쳐져도 알아채지 못하지만, 같은 함수를 호출하면
    그럴 수 없다. "조건을 맞췄다" 가 문서상의 주장이 아니라 코드로 확인된다.

(2) 두 버전을 같은 세션에서 연속 실행한다.

    필수 실험의 Q4_K_M 기록을 그대로 가져다 쓰지 않는다. VRAM 여유와 GPU
    상태가 세션마다 달라 속도·적재 비교가 오염되기 때문이다.

    부수 효과가 하나 있다. 이번 Q4_K_M 재실행값을 필수 실험값과 나란히 두면
    **같은 모델·같은 설정을 다시 돌렸을 때 얼마나 흔들리는지** 알 수 있다.
    실제로 이 점검에서 품질 점수가 6.65점 움직이는 것을 발견했고, 그 값이
    양자화 효과(3.25점)보다 커서 품질 결론을 "판정 불가" 로 제한했다.
    (docs/06_quantization.md 7절)

    uv run python src/run_quant.py

출력
    results/quant_raw.jsonl      양자화 비교 회차 (필수 실험 40회와 분리)
    results/quant_warmup.jsonl   워밍업 (집계 제외)
    results/errors.jsonl         호출 실패 (필수 실험과 같은 파일, model 로 구분)
"""

import argparse
import io
import os
import sys

# src/ 는 패키지가 아니므로 run_local 을 import 하려면 경로를 직접 넣어야 한다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ollama import Client

# 필수 실험 스크립트를 그대로 가져온다. OPTIONS·THINK·REPEATS·load_inputs·
# call·build_record·error_record·append 를 전부 재사용하므로, 이 파일에는
# "무엇을 비교할지" 만 남고 "어떻게 호출·기록할지" 는 한 곳에만 존재한다.
import run_local as base

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# 필수 실험 40회(local_raw.jsonl)와 파일을 분리한다. 선택 실습은 필수 항목이
# 아니므로 집계·채점·제출물에 섞이면 안 된다.
RAW = os.path.join(RESULTS, "quant_raw.jsonl")
WARMUP = os.path.join(RESULTS, "quant_warmup.jsonl")
ERRORS = os.path.join(RESULTS, "errors.jsonl")   # 오류는 필수 실험과 같은 파일

# 같은 베이스 모델의 두 양자화본. 순서는 4bit → 8bit.
# 순서를 고정하는 이유: 뒤에 도는 쪽이 앞의 잔여 VRAM 상태를 물려받을 수
# 있으므로, 재실행할 때마다 순서가 바뀌면 비교가 흔들린다.
PAIR = ["gemma4:e4b", "gemma4:e4b-it-q8_0"]


def already_done():
    """이미 기록된 회차 수. 덮어쓰기 사고를 막기 위한 확인용."""
    if not os.path.exists(RAW):
        return 0
    return sum(1 for _ in io.open(RAW, encoding="utf-8"))


def run_one(client, model, sysp, questions, repeats):
    """한 양자화본에 대해 콜드 시작 → 워밍업 1회 → 본 회차 20회를 수행한다.

    run_local.py 의 main() 과 같은 절차다. 절차가 같아야 두 실험의 측정값을
    나란히 놓을 수 있다.
    """
    print("=" * 66)
    print("모델 :", model)

    # keep_alive=0 은 "지금 메모리에서 내려라" 는 뜻이다. 이전 모델이 남아
    # 있으면 로딩 시간이 0초로 찍혀 콜드 로딩을 측정할 수 없다.
    client.generate(model=model, keep_alive=0)

    # ── 워밍업 1회 ────────────────────────────────────────────────
    # 첫 호출에는 모델을 디스크에서 VRAM으로 올리는 시간이 섞인다. 그대로
    # 집계하면 응답 시간 평균이 왜곡되므로 별도 파일로 뺀다. 동시에 이 회차가
    # VRAM 실측값(size_vram)과 CPU/GPU 적재 비율을 얻는 지점이기도 하다.
    try:
        el, r = base.call(client, model, sysp, base.WARMUP_PROMPT)
        rec = base.build_record(client, model, "WARMUP", "warmup", None, 0, True, el, r)
        rec["exercise"] = "quantization"
        base.append(WARMUP, rec)
        print("워밍업 : %.2f초 (로딩 %.2f초) | VRAM %s MiB | %s — 집계 제외"
              % (el, rec["load_s"], rec["vram_mib"], rec["processor"]))
    except Exception as e:
        # 워밍업이 실패해도 본 회차는 계속한다. 실패 사실만 기록한다.
        base.append(ERRORS, base.error_record(model, "WARMUP", 0, True, 0.0, e))
        print("워밍업 실패 : %s — 기록 후 계속" % type(e).__name__)

    # ── 본 회차 ──────────────────────────────────────────────────
    # 바깥이 반복 회차, 안쪽이 질문이다. Q1~Q10 을 한 바퀴 돈 뒤 다시 한 바퀴
    # 돈다. 호출마다 대화 이력을 초기화하므로 회차 순서가 응답에 영향을 주지는
    # 않는다. run_local.py 와 순서를 맞춰 두 실험의 기록 형태를 같게 하려는
    # 것이다.
    ok = 0
    attempts = 0
    for run in range(1, repeats + 1):
        for q in questions:
            attempts += 1
            try:
                el, r = base.call(client, model, sysp, q["question"])
            except Exception as e:
                # 예외를 좁게 나열하면 예상 못 한 하나에 40회가 중간에 멈춘다.
                # 넓게 잡아 기록하고 다음 회차로 넘어간다. 실패도 결과다.
                base.append(ERRORS, base.error_record(model, q["id"], run, False, 0.0, e))
                print("  [%s run%d] 실패 %s" % (q["id"], run, type(e).__name__))
                continue

            rec = base.build_record(client, model, q["id"], q["type"], q.get("grounded"),
                                    run, False, el, r)
            rec["exercise"] = "quantization"     # 필수 실험 기록과 구분하는 표시
            base.append(RAW, rec)                # 한 건씩 즉시 기록 — 중단돼도 남는다
            ok += 1

            # 측정 불가 값을 0 으로 채우지 않는다. build_record 가 None 을 넣고
            # gen_speed_note 에 사유를 남긴다.
            speed = ("%.1f tok/s" % rec["gen_tok_per_s"]) if rec["gen_tok_per_s"] else "계산 불가"
            print("  [%s run%d] %.2fs | in %d + out %d | %s | %s"
                  % (q["id"], run, el, rec["prompt_tok"], rec["out_tok"],
                     speed, rec["done_reason"]))

    # 다음 버전이 VRAM 을 온전히 쓰도록 내린다. 8bit 는 5,011 MiB 를 쓰므로
    # 4bit(3,077 MiB)가 남아 있으면 둘이 동시에 올라가지 못한다.
    client.generate(model=model, keep_alive=0)
    print("성공 %d / 시도 %d" % (ok, attempts))
    print()
    return ok, attempts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--repeats", type=int, default=base.REPEATS)
    ap.add_argument("--force", action="store_true", help="기록이 있어도 이어서 추가")
    args = ap.parse_args()

    # 실수로 두 번 실행해 40회가 80회로 불어나는 것을 막는다. 재실험이라면
    # --force 로 명시적으로 이어 붙이고, 재실험분임을 문서에 표시한다.
    done = already_done()
    if done and not args.force:
        sys.exit("이미 양자화 비교 기록이 %d건 있습니다.\n"
                 "재실험이라면 --force 로 이어 붙이고 재실험분임을 별도로 표시하십시오.\n"
                 "처음부터 다시 하려면 %s 를 먼저 정리하십시오." % (done, RAW))

    # 지시문·정책 문서·질문 10개를 필수 실험과 같은 함수로 읽는다.
    # questions.json 은 최초 확정 커밋(7800589) 이후 수정하지 않았다.
    sysp, questions = base.load_inputs()

    # timeout=600 은 예방 조치였다. 실행 전에는 디스크 11 GB 인 8bit 가 VRAM
    # 8,151 MiB 를 넘겨 CPU 로 분할되고 그만큼 느려질 것으로 예상했다.
    # 실제로는 두 버전 모두 100% GPU 로 적재됐다 — Gemma 4 E계열의 Per-Layer
    # Embeddings 테이블이 VRAM 에 상주하지 않기 때문이다. 예측이 빗나갔지만
    # 여유는 그대로 둔다. (docs/06_quantization.md 4절)
    client = Client(host=args.host, timeout=600)

    print("선택 실습 A — Quantization 비교")
    print("비교 대상 : %s" % " vs ".join(PAIR))
    print("질문      : %d개 × %d회 × %d버전 = %d회"
          % (len(questions), args.repeats, len(PAIR),
             len(questions) * args.repeats * len(PAIR)))
    # 생성 설정을 화면에도 찍는다. 결과 파일의 options 필드와 대조해 실행
    # 조건이 실제로 동일했는지 사후에 확인할 수 있어야 한다.
    print("생성 설정 : %s, think=%s  (필수 실험과 동일)" % (base.OPTIONS, base.THINK))
    print()

    for model in PAIR:
        run_one(client, model, sysp, questions, args.repeats)

    print("본 회차 :", RAW)
    print("워밍업  :", WARMUP)
    print()
    # 이 스크립트는 측정만 한다. 채점은 사람이 응답 전문을 읽고 부여하며,
    # aggregate_quant.py 의 채점표에 근거와 함께 기록돼 있다.
    print("집계    : uv run python src/aggregate_quant.py")


if __name__ == "__main__":
    main()
