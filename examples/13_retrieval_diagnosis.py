"""선택 실습 B STEP 6b — 검색이 왜 틀렸는지 진단한다.

    uv run python examples/13_retrieval_diagnosis.py

12_similarity_is_not_truth.py 2부에서 절 단위 검색이 3문항 중 1개만 맞혔다.
Q5 와 Q6 은 서로 답이 뒤바뀌었다.

    Q5 "혼자 가는데 추가 요금이 붙나요?"      -> 4. 불포함 사항   (정답 2. 요금 기준)
    Q6 "저녁 식사는 요금에 포함되어 있나요?"   -> 2. 요금 기준     (정답 4. 불포함 사항)

원인 가설 세 가지를 차례로 확인한다.

    H1  절이 임베딩 모델의 128토큰 한계를 넘어 뒷부분이 잘렸다
    H2  절 단위가 너무 커서 한 조각에 여러 주제가 섞였다
    H3  '포함 / 불포함' 같은 부정 표현을 약하게 본다

이 실습의 요구사항은 12번까지로 충족된다. 이 파일은 틀린 결과를 그대로
두지 않기 위한 추가 확인이다.
"""

import io
import os
import re

from sentence_transformers import SentenceTransformer, util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL = "jhgan/ko-sroberta-multitask"     # 전 단계와 같은 모델·설정
model = SentenceTransformer(MODEL)
tok = model.tokenizer
LIMIT = model.max_seq_length

raw = io.open(os.path.join(ROOT, "data", "policy.md"), encoding="utf-8").read()

QUESTIONS = [
    ("Q1", "출발 25일 전에 취소하면 수수료가 얼마인가요?", "5. 취소 수수료 규정"),
    ("Q5", "혼자 가는데 추가 요금이 붙나요?", "2. 요금 기준"),
    ("Q6", "저녁 식사는 요금에 포함되어 있나요?", "4. 불포함 사항"),
]


def sections():
    out = []
    for block in re.split(r"\n(?=## )", raw):
        block = block.strip()
        if not block.startswith("## "):
            continue
        title = block.splitlines()[0].replace("## ", "").strip()
        out.append((title, " ".join(block.split())))
    return out


SEC = sections()

# ══ H1 — 128토큰 한계를 넘는 절이 있는가 ═══════════════════════
print("H1. 절이 임베딩 모델의 %d토큰 한계를 넘는가" % LIMIT)
print("=" * 72)
print("%-24s %8s %8s  %s" % ("절", "토큰", "한계", "상태"))
print("-" * 72)
truncated = 0
for title, body in SEC:
    n = len(tok.encode(body))
    over = n > LIMIT
    truncated += over
    print("%-24s %8d %8d  %s" % (title[:22], n, LIMIT, "잘림" if over else "OK"))
print()
print("=> 잘린 절 %d개 / 전체 %d개" % (truncated, len(SEC)))
if truncated:
    print("   잘린 절은 뒷부분 내용이 임베딩에 반영되지 않는다.")
    print("   H1 이 원인 중 하나다.")
else:
    print("   모든 절이 한계 안에 들어간다. H1 은 원인이 아니다.")
print()
print()

# ══ H3 — 전체 순위를 본다 ═════════════════════════════════════
print("H3. 1등만 보지 말고 전체 순위를 보자 (절 단위)")
print("=" * 72)
vecs = model.encode([b for _, b in SEC])
for qid, q, answer in QUESTIONS:
    scores = util.cos_sim(model.encode(q), vecs).numpy()[0]
    order = scores.argsort()[::-1]
    rank = [i for i, idx in enumerate(order, 1) if SEC[idx][0] == answer]
    print()
    print("%s  %s" % (qid, q))
    print("   정답 절: %s  ->  실제 순위 %s위" % (answer, rank[0] if rank else "?"))
    for r, idx in enumerate(order[:3], 1):
        mark = " <- 정답" if SEC[idx][0] == answer else ""
        print("     %d위 %+.4f  %s%s" % (r, scores[idx], SEC[idx][0], mark))
print()
print("=> 정답이 1등이 아니어도 2~3위 안에 있다면, 상위 k개를 모두 넘기는")
print("   방식(top-k)으로 실용성을 확보할 수 있다. RAG 가 보통 3~5개를")
print("   가져오는 이유다.")
print()
print()

# ══ H2 — 문장 단위로 쪼개면 나아지는가 ════════════════════════
print("H2. 절 대신 문장 단위로 쪼개면 나아지는가")
print("=" * 72)

# 절 제목을 각 문장 앞에 붙인다. 문장만 떼면 무슨 조항인지 맥락이 사라진다.
fine = []
for title, body in SEC:
    text = body[len("## " + title):].strip() if body.startswith("## ") else body
    for part in re.split(r"(?<=[.다])\s+|(?=- )", text):
        part = part.strip(" -")
        if len(part) < 10:
            continue
        fine.append((title, "%s: %s" % (title, part)))

print("절 %d개 -> 문장 %d개로 세분화" % (len(SEC), len(fine)))
print("(각 문장 앞에 절 제목을 붙여 맥락을 유지한다)")
print()

fine_vecs = model.encode([t for _, t in fine])
hit_sec = hit_fine = 0
print("%-5s %-26s %-26s" % ("질문", "절 단위 1등", "문장 단위 1등"))
print("-" * 72)
for qid, q, answer in QUESTIONS:
    qv = model.encode(q)

    s1 = util.cos_sim(qv, vecs).numpy()[0]
    top_sec = SEC[int(s1.argmax())][0]

    s2 = util.cos_sim(qv, fine_vecs).numpy()[0]
    top_fine = fine[int(s2.argmax())][0]

    hit_sec += top_sec == answer
    hit_fine += top_fine == answer
    print("%-5s %-26s %-26s"
          % (qid,
             "%s %s" % ("O" if top_sec == answer else "X", top_sec[:22]),
             "%s %s" % ("O" if top_fine == answer else "X", top_fine[:22])))

print("-" * 72)
print("%-5s %-26s %-26s" % ("정답", "%d / %d" % (hit_sec, len(QUESTIONS)),
                            "%d / %d" % (hit_fine, len(QUESTIONS))))
print()
print("가장 가까운 문장 (문장 단위)")
print("-" * 72)
for qid, q, answer in QUESTIONS:
    s2 = util.cos_sim(model.encode(q), fine_vecs).numpy()[0]
    best = int(s2.argmax())
    print("%s %+.4f  %s" % (qid, s2[best], fine[best][1][:60]))
print()
print("=" * 72)
print("결론은 실행 결과를 보고 적는다. 미리 정해두지 않는다.")
