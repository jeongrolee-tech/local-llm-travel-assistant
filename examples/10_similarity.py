"""선택 실습 B STEP 4 — Semantic Similarity. 문장들을 서로 비교한다.

    uv run python examples/10_similarity.py

발제문 흐름의 마지막 화살표.

    Sentence -> Embedding Vector -> Cosine Similarity -> Semantic Similarity
                                                         ^^^^^^^^^^^^^^^^^^^

발제문이 지정한 세 문장으로 먼저 비교한 뒤, 임베딩이 단순 단어 겹침과
무엇이 다른지 보기 위해 두 문장을 더 넣는다.

    S4  소파가 고양이 위에 있다     단어는 거의 같은데 의미가 뒤집힘
    S5  야옹이가 쿠션에 누워 있다   단어는 안 겹치는데 의미가 비슷

비교 기준으로 어휘 겹침(Jaccard)을 같이 계산한다. 키워드 매칭이라면 이 값이
높을수록 비슷하다고 판정할 것이다. 임베딩이 그와 얼마나 다르게 보는지가
이 실습의 관찰 지점이다.
"""

import numpy as np
from sentence_transformers import SentenceTransformer, util

MODEL = "jhgan/ko-sroberta-multitask"     # 전 단계와 같은 모델·설정
model = SentenceTransformer(MODEL)

S = [
    "고양이가 소파 위에 있다",        # S1  기준
    "소파 위에 고양이가 앉아 있다",    # S2  발제문 지정 - 의미 같음
    "오늘 주식시장이 상승했다",        # S3  발제문 지정 - 무관
    "소파가 고양이 위에 있다",        # S4  단어 거의 동일, 의미 뒤집힘
    "야옹이가 쿠션에 누워 있다",      # S5  단어 안 겹침, 의미 비슷
]

V = model.encode(S)
M = util.cos_sim(V, V).numpy()            # 5x5 유사도 행렬


def jaccard(a, b):
    """어휘 겹침. 키워드 매칭이 보는 값과 같은 성격이다."""
    A, B = set(a.split()), set(b.split())
    return len(A & B) / len(A | B)


print("임베딩 모델 :", MODEL, "(전 단계와 동일)")
print()
for i, s in enumerate(S, 1):
    print("  S%d  %s" % (i, s))
print()

# ── 발제문 지정 3문장 ────────────────────────────────────────────
print("발제문 지정 3문장")
print("=" * 66)
print("%-34s %-10s %s" % ("비교", "유사도", "판정"))
print("-" * 66)
for i, j in [(0, 1), (0, 2), (1, 2)]:
    v = M[i][j]
    verdict = "의미 가까움" if v >= 0.7 else ("애매" if v >= 0.4 else "무관")
    print("%-34s %+.4f     %s" % ("S%d vs S%d" % (i + 1, j + 1), v, verdict))
print()
print("=> S1-S2 는 어순과 서술어가 다른데도 높다. 글자가 아니라 의미를 본다.")
print("   S3 는 둘 다와 낮다. 주제가 다르면 멀어진다.")
print()

# ── 전체 행렬 ────────────────────────────────────────────────────
print("전체 유사도 행렬")
print("=" * 66)
print("%6s" % "", end="")
for j in range(len(S)):
    print("%9s" % ("S%d" % (j + 1)), end="")
print()
for i in range(len(S)):
    print("%6s" % ("S%d" % (i + 1)), end="")
    for j in range(len(S)):
        print("%9.4f" % M[i][j], end="")
    print()
print()
print("=> 대각선은 자기 자신이므로 1.0000 이다.")
print()

# ── 임베딩 vs 어휘 겹침 ──────────────────────────────────────────
print("임베딩은 단어 겹침과 다르게 본다  (기준 = S1)")
print("=" * 66)
print("%-26s %-11s %-11s %s" % ("비교 문장", "어휘 겹침", "임베딩", "무엇을 말하나"))
print("-" * 66)
rows = [
    (1, "어순·서술어 다름"),
    (2, "주제가 다름"),
    (3, "단어는 같고 의미 반대"),
    (4, "단어 다르고 의미 비슷"),
]
for idx, label in rows:
    print("%-26s %9.3f   %+9.4f   %s"
          % (S[idx][:24], jaccard(S[0], S[idx]), M[0][idx], label))
print()
print("관찰 지점")
print("-" * 66)
print("S5  어휘 겹침 %.3f 인데 임베딩 %+.4f."
      % (jaccard(S[0], S[4]), M[0][4]))
print("    '고양이/야옹이', '소파/쿠션' 처럼 겹치는 단어가 거의 없는데도 가깝다.")
print("    키워드 매칭으로는 절대 못 찾는 문장을 임베딩은 찾아낸다.")
print("    RAG 가 키워드 검색 대신 임베딩을 쓰는 이유다.")
print()
print("S4  어휘 겹침 %.3f 로 가장 높은데 임베딩도 %+.4f 로 높다."
      % (jaccard(S[0], S[3]), M[0][3]))
print("    그런데 S4 는 '소파가 고양이 위에' 로 위아래가 뒤집힌 문장이다.")
print("    임베딩은 무엇에 대한 말인지는 잘 잡지만, 주어와 목적어가 바뀐 것 같은")
print("    논리 구조 차이는 약하게 본다. 이 한계가 STEP 6 의 핵심이다.")
