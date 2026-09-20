"""선택 실습 B STEP 3 — 코사인 유사도를 손으로 계산해 본다.

    uv run python examples/09_cosine.py

발제문 흐름의 세 번째 화살표.

    Sentence -> Embedding Vector -> Cosine Similarity -> Semantic Similarity
                                    ^^^^^^^^^^^^^^^^^

공식
    cos(A, B) = (A · B) / (|A| x |B|)

    A · B : 내적. 같은 자리끼리 곱해서 전부 더한 값
    |A|   : 크기(norm). 제곱합의 제곱근

확인할 것
    (1) 공식대로 직접 계산한 값과 라이브러리 값이 같다
    (2) 왜 거리가 아니라 각도를 쓰는가
    (3) 정규화하면 내적 자체가 코사인이 된다
"""

import numpy as np
from sentence_transformers import SentenceTransformer, util

MODEL = "jhgan/ko-sroberta-multitask"
model = SentenceTransformer(MODEL)

a_text = "고양이가 소파 위에 있다"
b_text = "소파 위에 고양이가 앉아 있다"

A, B = model.encode([a_text, b_text])

print("A :", a_text)
print("B :", b_text)
print()

# ── (1) 공식대로 손계산 ──────────────────────────────────────────
# 내적: 768개 자리를 각각 곱해서 전부 더한다.
dot = float(np.dot(A, B))

# 크기: 각 원소를 제곱해 더하고 제곱근. np.linalg.norm 이 이걸 한다.
norm_a = float(np.linalg.norm(A))
norm_b = float(np.linalg.norm(B))

cos_manual = dot / (norm_a * norm_b)

print("손계산")
print("-" * 58)
print("A · B  (내적)        = %+10.4f" % dot)
print("|A|    (A 의 크기)   = %10.4f" % norm_a)
print("|B|    (B 의 크기)   = %10.4f" % norm_b)
print("-" * 58)
print("cos    = %+.4f / (%.4f x %.4f)" % (dot, norm_a, norm_b))
print("       = %+.6f" % cos_manual)
print()

# ── 라이브러리 값과 대조 ─────────────────────────────────────────
# util.cos_sim 은 같은 계산을 한다. 값이 같아야 이해가 맞은 것이다.
cos_lib = float(util.cos_sim(A, B))
print("라이브러리 util.cos_sim = %+.6f" % cos_lib)
print("차이                    = %.2e  -> %s"
      % (abs(cos_manual - cos_lib),
         "같은 계산이다" if abs(cos_manual - cos_lib) < 1e-5 else "다르다"))
print()

# ── (2) 왜 거리가 아니라 각도인가 ────────────────────────────────
# B 를 2배로 늘려 본다. 방향은 그대로고 길이만 2배가 된다.
B2 = B * 2

print("왜 유클리드 거리가 아니라 코사인인가")
print("-" * 58)
print("B 를 2배로 늘린 B2 를 만든다. 방향은 그대로, 길이만 2배.")
print()
print("%-22s %-12s %s" % ("", "A vs B", "A vs B2"))
print("%-22s %-12.4f %.4f"
      % ("유클리드 거리", np.linalg.norm(A - B), np.linalg.norm(A - B2)))
print("%-22s %-12.6f %.6f"
      % ("코사인 유사도", cos_manual, float(util.cos_sim(A, B2))))
print()
print("=> 거리는 크게 변했지만 코사인은 그대로다.")
print("   코사인은 길이를 무시하고 방향만 본다. 문장이 길거나 표현이 세서")
print("   벡터가 커져도 '의미가 같으면 같게' 나와야 하므로 이 성질이 필요하다.")
print()

# ── (3) 정규화하면 내적이 곧 코사인 ──────────────────────────────
# 크기를 1 로 맞추면 분모가 1 x 1 = 1 이 되어 내적만 남는다.
An = A / np.linalg.norm(A)
Bn = B / np.linalg.norm(B)

print("정규화(길이를 1 로 맞추기)")
print("-" * 58)
print("정규화 후 |A| = %.6f, |B| = %.6f" % (np.linalg.norm(An), np.linalg.norm(Bn)))
print("정규화한 두 벡터의 내적 = %+.6f" % float(np.dot(An, Bn)))
print()
print("=> 분모가 1 이 되므로 내적 자체가 코사인 유사도다.")
print("   벡터 DB 가 미리 정규화해 두는 이유다 — 검색할 때마다 나눗셈을")
print("   반복하지 않고 내적만 하면 된다.")
print()

# ── 값의 범위 ────────────────────────────────────────────────────
print("코사인 값의 범위")
print("-" * 58)
print("  +1  방향이 완전히 같음")
print("   0  직각. 서로 무관")
print("  -1  정반대")
print()
print("이번 두 문장 : %+.4f" % cos_manual)
print()
print("주의 — 이 값은 '의미가 얼마나 가까운가' 이지 '내용이 맞는가' 가 아니다.")
print("       STEP 6 에서 이 구분을 숫자로 확인한다.")
