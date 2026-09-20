"""선택 실습 B STEP 6 — 유사도는 사실성 점수가 아니다.

    uv run python examples/12_similarity_is_not_truth.py

발제문 요구: "유사도 값을 답변의 사실성 점수로 해석하지 않습니다."

1부 — 유사도로 못 하는 것
    data/policy.md 의 실제 문장을 기준으로, 숫자만 바꾼 거짓 문장과 뜻을
    뒤집은 문장을 넣어 본다. 사실 여부와 유사도가 따로 논다는 것을 확인한다.

2부 — 유사도로 되는 것
    정책 문서를 절 단위로 쪼개고 질문과의 유사도로 해당 절을 찾아낸다.
    RAG 의 R(Retrieval)에 해당하는 동작이며, 이쪽은 잘 작동한다.

결론은 "임베딩이 쓸모없다" 가 아니라 "어디에 쓰고 어디에 쓰지 않는가" 다.
"""

import io
import os
import re

from sentence_transformers import SentenceTransformer, util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL = "jhgan/ko-sroberta-multitask"     # 전 단계와 같은 모델·설정
model = SentenceTransformer(MODEL)


def sim(a, b):
    return float(util.cos_sim(model.encode(a), model.encode(b)))


# ══ 1부 — 유사도로 못 하는 것 ═══════════════════════════════════
print("1부. 유사도는 사실 여부를 구분하지 못한다")
print("=" * 72)

CASES = [
    {
        "기준": "일본은 대한민국 여권 소지자에 한해 90일 이내 관광 목적 체류 시 비자가 면제됩니다.",
        "출처": "policy.md 8절",
        "변형": [
            ("참", "한국 여권이 있으면 관광 목적으로 일본에 90일까지 비자 없이 갈 수 있습니다."),
            ("거짓", "일본은 대한민국 여권 소지자에 한해 30일 이내 관광 목적 체류 시 비자가 면제됩니다."),
            ("거짓", "일본은 대한민국 여권 소지자도 관광 목적 체류 시 비자가 필요합니다."),
            ("무관", "조식 3회가 요금에 포함되어 있습니다."),
        ],
    },
    {
        "기준": "출발 29일 전부터 20일 전 사이에 취소하면 요금의 10%가 부과됩니다.",
        "출처": "policy.md 5절",
        "변형": [
            ("참", "출발 25일 전에 취소하시면 취소 수수료로 요금의 10%를 내셔야 합니다."),
            ("거짓", "출발 29일 전부터 20일 전 사이에 취소하면 요금의 50%가 부과됩니다."),
            ("거짓", "출발 29일 전부터 20일 전 사이에 취소하면 수수료가 없습니다."),
            ("무관", "인천국제공항에서 출발하는 오사카 3박 4일 패키지입니다."),
        ],
    },
]

for case in CASES:
    print()
    print("기준 문장 (%s):" % case["출처"])
    print("  %s" % case["기준"])
    print()
    print("  %-6s %-9s %s" % ("사실", "유사도", "문장"))
    print("  " + "-" * 68)
    scored = [(kind, sim(case["기준"], t), t) for kind, t in case["변형"]]
    for kind, v, t in scored:
        mark = {"참": "O", "거짓": "X", "무관": "-"}[kind]
        print("  %-6s %+.4f    %s" % ("%s %s" % (mark, kind), v, t[:44]))

    true_s = [v for k, v, _ in scored if k == "참"]
    false_s = [v for k, v, _ in scored if k == "거짓"]
    print()
    print("  참 문장 최고 %+.4f / 거짓 문장 최고 %+.4f -> %s"
          % (max(true_s), max(false_s),
             "거짓이 더 높다" if max(false_s) > max(true_s) else "참이 더 높다"))

print()
print("=" * 72)
print("관찰")
print("-" * 72)
print("숫자만 바꾼 거짓 문장은 기준과 거의 같은 값을 받는다. 글자가 두 자")
print("달라졌을 뿐이고, 임베딩이 보는 '무엇에 대한 말인가' 는 바뀌지 않았다.")
print()
print("본 실험에서 Kanana 2 3B 는 문서에 없는 '3천만원' 과 '90일 무비자' 를")
print("지어내 P4 에서 탈락했다. 만약 유사도로 자동 채점했다면 그 답변들도")
print("원문과 높은 유사도를 받아 통과했을 것이다.")
print()
print("=> 그래서 이 프로젝트는 data/rubric.md 로 사람이 채점했다.")
print("   유사도는 '비슷한가' 를 재는 자이지 '맞는가' 를 재는 자가 아니다.")
print()
print()

# ══ 2부 — 유사도로 되는 것 ═════════════════════════════════════
print("2부. 유사도가 잘하는 일 — 어느 조항을 봐야 하는지 찾기")
print("=" * 72)

# policy.md 를 '## ' 기준으로 절 단위로 쪼갠다. RAG 의 청킹에 해당한다.
raw = io.open(os.path.join(ROOT, "data", "policy.md"), encoding="utf-8").read()
chunks = []
for block in re.split(r"\n(?=## )", raw):
    block = block.strip()
    if not block.startswith("## "):
        continue
    title = block.splitlines()[0].replace("## ", "").strip()
    chunks.append((title, " ".join(block.split())))

print("policy.md 를 절 단위로 %d개 조각으로 나눴다." % len(chunks))
print("(전문 873토큰은 임베딩 모델의 128토큰 한계를 넘으므로 쪼개야 한다)")
print()

chunk_vecs = model.encode([c[1] for c in chunks])

QUESTIONS = [
    ("Q1", "출발 25일 전에 취소하면 수수료가 얼마인가요?", "5. 취소 수수료 규정"),
    ("Q5", "혼자 가는데 추가 요금이 붙나요?", "2. 요금 기준"),
    ("Q6", "저녁 식사는 요금에 포함되어 있나요?", "4. 불포함 사항"),
    ("Q9", "여행자보험 상해 보장 한도가 얼마인가요?", "(문서에 없음)"),
]

print("%-5s %-34s %-22s %s" % ("질문", "가장 가까운 절", "유사도", "정답 절"))
print("-" * 72)
hit = 0
for qid, q, answer_section in QUESTIONS:
    scores = util.cos_sim(model.encode(q), chunk_vecs).numpy()[0]
    best = int(scores.argmax())
    title = chunks[best][0]
    ok = title == answer_section
    hit += ok
    mark = "O" if ok else ("-" if answer_section.startswith("(") else "X")
    print("%-5s %-34s %+.4f %-12s %s %s"
          % (qid, title[:32], scores[best], "", mark, answer_section))

print()
print("정답 절을 찾은 경우: %d / %d (Q9 는 문서에 답이 없어 제외)"
      % (hit, len(QUESTIONS) - 1))
print()
print("Q9 를 보라. 문서에 보험 조항이 없는데도 무언가를 '가장 가깝다' 고 고른다.")
print("검색은 항상 1등을 내놓기 때문이다. 그래서 실제 RAG 는 유사도가 일정")
print("기준 미만이면 '관련 문서 없음' 으로 처리하는 문턱값을 둔다.")
print()
print("=" * 72)
print("정리")
print("-" * 72)
print("%-44s %s" % ("어느 조항을 봐야 하는지 찾기", "유사도로 된다"))
print("%-44s %s" % ("그 조항에 비슷한 말인지 재기", "유사도로 된다"))
print("%-44s %s" % ("그 답변의 내용이 맞는지 판정하기", "유사도로 안 된다"))
print()
print("RAG 는 앞의 둘만 유사도에 맡기고, 답변이 맞는지는 생성 모델과")
print("사람의 검수에 맡긴다. 본 실험이 루브릭으로 채점한 것과 같은 이유다.")
