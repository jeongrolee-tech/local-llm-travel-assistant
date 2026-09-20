"""선택 실습 B STEP 5 — 토큰 ID 와 임베딩을 구분한다.

    uv run python examples/11_token_vs_embedding.py

발제문 요구: "생성 모델의 토큰 ID와 문장 비교용 임베딩을 구분합니다."

둘 다 "문장을 숫자로 바꾼 것" 이라 헷갈리기 쉽다. 실제로는 다른 물건이다.

    토큰 ID   문장을 사전의 몇 번째 칸인지로 바꾼 정수 목록.
              길이에 비례해 늘어나며, 숫자의 크고 작음에 의미가 없다.
              모델마다 사전이 다르므로 ID 도 호환되지 않는다.

    임베딩     문장 전체를 의미 공간의 좌표 하나로 바꾼 실수 벡터.
              길이와 무관하게 항상 같은 차원이며, 방향에 의미가 있다.

확인할 것
    (1) 같은 문장의 토큰 ID 와 임베딩이 어떻게 다르게 생겼는가
    (2) 인접한 토큰 ID 가 서로 아무 관계도 없다는 것
    (3) 모델이 다르면 토큰 수 자체가 다르다는 것
"""

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "jhgan/ko-sroberta-multitask"
model = SentenceTransformer(MODEL)
tok = model.tokenizer

TEXT = "고양이가 소파 위에 있다"

# ── (1) 같은 문장, 두 가지 표현 ──────────────────────────────────
ids = tok.encode(TEXT)                 # 정수 목록
pieces = tok.convert_ids_to_tokens(ids)
vec = model.encode(TEXT)               # 실수 768개

print("문장 :", TEXT)
print()
print("[A] 토큰 ID — 문장을 사전 번호로 바꾼 것")
print("-" * 66)
print("개수 :", len(ids))
print("ID   :", ids)
print("조각 :", pieces)
print()
print("[B] 임베딩 — 문장을 의미 좌표로 바꾼 것")
print("-" * 66)
print("개수 :", vec.shape[0])
print("앞 6개:", ", ".join("%+.4f" % x for x in vec[:6]), "...")
print()

print("%-16s %-24s %s" % ("", "토큰 ID", "임베딩"))
print("-" * 66)
print("%-16s %-24s %s" % ("자료형", "정수(int)", "실수(float)"))
print("%-16s %-24s %s" % ("개수", "%d개 (문장 길이에 비례)" % len(ids), "768개 (항상 고정)"))
print("%-16s %-24s %s" % ("뜻", "사전의 몇 번째 칸", "의미 공간의 좌표"))
print("%-16s %-24s %s" % ("크기 비교", "의미 없음", "방향에 의미 있음"))
print("%-16s %-24s %s" % ("만드는 주체", "토크나이저", "모델 전체"))
print()

# ── (2) 인접한 ID 는 아무 관계가 없다 ────────────────────────────
# 임베딩이라면 가까운 좌표는 비슷한 의미다. 토큰 ID 는 그렇지 않다.
base = ids[1]                          # 첫 실질 토큰
print("인접한 토큰 ID 를 사람 말로 되돌리면")
print("-" * 66)
for d in range(-2, 3):
    i = base + d
    word = tok.convert_ids_to_tokens([i])[0]
    mark = "  <- 원래 토큰" if d == 0 else ""
    print("  ID %-8d %s%s" % (i, word, mark))
print()
print("=> ID 가 1 차이 나도 뜻은 전혀 상관없다. 사전에 나란히 적혔을 뿐이다.")
print("   따라서 토큰 ID 끼리 빼거나 평균 내는 것은 아무 의미가 없다.")
print("   임베딩은 반대다 — 가까운 좌표는 실제로 비슷한 의미다 (STEP 4).")
print()

# ── (3) 모델이 다르면 토큰 수가 다르다 ──────────────────────────
# 본 실험에서 잰 값이다 (docs/02_model_comparison.md 3절).
# 같은 한국어 문서를 각 모델이 몇 토큰으로 표현했는가.
print("모델마다 사전이 다르므로 토큰 수도 ID 도 다르다")
print("-" * 66)
print("%-28s %10s %s" % ("모델", "토큰 수", "Kanana 대비"))
measured = [("Kanana 2 3B (한국어 특화)", 811, 0.0),
            ("Qwen3.5", 980, 20.8),
            ("Gemma 4", 1072, 32.1)]
for name, n, pct in measured:
    print("%-28s %10d %s" % (name, n, "기준" if pct == 0 else "+%.1f%%" % pct))
print()
print("이 임베딩 모델(%s)" % MODEL)
print("  같은 문장 '%s' -> %d 토큰" % (TEXT, len(ids)))
print()
print("=> 토큰 수가 적을수록 같은 내용을 더 적은 토큰으로 표현한 것이므로")
print("   토크나이저 효율이 좋다. 한국어 특화 모델인 Kanana 가 가장 효율적이었다.")
print("   다만 그 효율이 답변 품질로 이어지지는 않았다 (Kanana 는 P4 탈락).")
print()

# ── 정리 ────────────────────────────────────────────────────────
print("정리 — 어디에 무엇을 쓰는가")
print("=" * 66)
print("토큰 ID  생성 모델에 입력을 넣을 때. 답변을 만드는 데 쓴다.")
print("         본 실험의 prompt_tok / out_tok 이 이 토큰의 개수다.")
print()
print("임베딩    문장끼리 비교·검색할 때. 답변을 만들지 않는다.")
print("         RAG 의 검색 단계가 이것을 쓴다 (README 8-3).")
print()
print("주의 — 생성 모델 내부에도 토큰별 임베딩이 있지만, 그것은 토큰 단위이고")
print("       문장 비교용으로 학습된 것이 아니다. 그래서 발제문이 문장 비교에는")
print("       Sentence Transformers 를 쓰도록 지정했다.")
