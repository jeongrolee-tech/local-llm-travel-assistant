"""컨텍스트 예산 측정 — 지시문과 정책 문서가 몇 토큰을 차지하는가.

README 8-3(문서 주입 방식)과 docs/05_selection_report.md 권고 6에 쓰인 수치를
재현한다. "문서가 컨텍스트에 들어가니 RAG 가 필요 없다" 는 판단의 근거이므로,
추정이 아니라 실제 토크나이저로 재야 한다.

    uv run python src/measure_context.py

측정 방법
    num_predict=1 로 호출해 응답의 prompt_eval_count 를 읽는다. 모델이 실제로
    센 입력 토큰 수이므로 별도 토크나이저를 설치할 필요가 없고, 본 실험이 기록한
    prompt_tok 과 같은 값이다.

    구성 요소별 기여분은 차이로 구한다.
        (a) 빈 지시문 + 질문          -> 채팅 템플릿 + 질문
        (b) 지시문(문서 제외) + 질문  -> (b) - (a) = 지시문
        (c) 지시문(문서 포함) + 질문  -> (c) - (b) = 정책 문서

주의
    토큰 수는 모델마다 다르다. 기본값은 최종 선정 모델인 gemma4:e4b 이며,
    --model 로 바꿀 수 있다. 한국어 토크나이저 효율 차이 때문에 같은 문서라도
    Kanana 는 더 적게, Gemma 는 더 많이 센다 (docs/02_model_comparison.md 3절).
"""

import argparse
import io
import os

from ollama import Client

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 본 실험과 같은 설정. num_predict=1 은 측정용이며 본 실험은 미설정이다.
OPTIONS = {"temperature": 0.2, "num_ctx": 4096, "num_predict": 1}

# 답변이 잘리지 않도록 확보할 여유. 본 실험 최대 출력이 119토큰이었으므로
# 512 는 넉넉한 값이다.
ANSWER_HEADROOM = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--ctx", type=int, default=OPTIONS["num_ctx"])
    args = ap.parse_args()

    policy = io.open(os.path.join(ROOT, "data", "policy.md"), encoding="utf-8").read()
    tmpl = io.open(os.path.join(ROOT, "data", "system_prompt.txt"), encoding="utf-8").read()
    question = "출발 25일 전에 취소하면 수수료가 얼마인가요?"      # 본 실험 Q1

    client = Client(host=args.host, timeout=300)

    def count(system):
        r = client.chat(model=args.model, think=False, options=OPTIONS,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": question}])
        return r.prompt_eval_count

    bare = count("")                                    # 채팅 템플릿 + 질문
    no_doc = count(tmpl.replace("{policy}", ""))        # + 지시문
    full = count(tmpl.replace("{policy}", policy))      # + 정책 문서

    doc = full - no_doc
    instructions = no_doc - bare
    fixed = no_doc                                      # 문서를 뺀 고정 오버헤드

    print("컨텍스트 예산 측정 — %s, 질문 Q1 기준" % args.model)
    print("=" * 62)
    print("채팅 템플릿 + 질문            %5d 토큰" % bare)
    print("시스템 지시문 (규칙 5개)      %5d 토큰" % instructions)
    print("정책 문서 policy.md 전문      %5d 토큰   <- 전체의 %.0f%%"
          % (doc, doc / full * 100))
    print("-" * 62)
    print("입력 합계                     %5d 토큰" % full)
    print()
    print("num_ctx=%d 대비 %.0f%% 사용, 답변 여유 %d 토큰"
          % (args.ctx, full / args.ctx * 100, args.ctx - full))
    print()

    # ── RAG 전환 문턱 ────────────────────────────────────────────
    # 상품이 늘면 정책 문서가 그만큼 커진다고 가정한다. 상품 간 공통 약관이
    # 있으면 실제 증가폭은 이보다 작으므로, 아래는 보수적인 상한이다.
    print("상품 수별 필요 토큰 (문서 %d토큰씩 증가, 답변 여유 %d 확보)"
          % (doc, ANSWER_HEADROOM))
    print("=" * 62)
    print("%-10s %10s   %-8s %-8s %-8s" % ("상품 수", "필요 토큰", "ctx4096", "ctx8192", "ctx32768"))
    print("-" * 62)
    for n in (1, 2, 3, 4, 5, 10, 20, 50):
        need = fixed + doc * n + ANSWER_HEADROOM
        cells = tuple("OK" if need <= c else "초과" for c in (4096, 8192, 32768))
        print("%-10d %10d   %-8s %-8s %-8s" % (n, need, cells[0], cells[1], cells[2]))

    limit = (4096 - fixed - ANSWER_HEADROOM) // doc
    print()
    print("num_ctx=4096 에서 수용 가능한 상품 수: %d개" % limit)
    print()
    print("주의 — 이 장비에서는 num_ctx 확대가 해법이 되지 못한다.")
    print("      num_ctx=8192 에서 qwen3.5:9b 가 CPU 분할되고 속도가 11% 떨어졌다.")
    print("      (README 9절 #8). 상품 %d개를 넘으면 RAG 전환을 검토한다." % limit)


if __name__ == "__main__":
    main()
