"""선택 실습 B STEP 2 — 문장 하나가 벡터가 되는 것을 눈으로 본다.

    uv run python examples/08_embedding.py

발제문의 흐름 중 첫 화살표에 해당한다.

    Sentence -> Embedding Vector -> Cosine Similarity -> Semantic Similarity
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

확인할 것 세 가지
    (1) 문장 길이가 달라도 벡터 길이는 항상 같다
    (2) 그 숫자들이 대체 어떻게 생겼는가
    (3) 이 모델이 한 번에 읽을 수 있는 길이에 한계가 있다
"""

from sentence_transformers import SentenceTransformer

# STEP 1 에서 받아둔 모델. 한국어 문장 유사도 과제로 학습된 모델이다.
# 발제문 요구: "같은 임베딩 모델·설정으로 비교" — 이 실습 전체에서 이 모델만 쓴다.
MODEL = "jhgan/ko-sroberta-multitask"

model = SentenceTransformer(MODEL)
dim = model.get_embedding_dimension()

print("임베딩 모델 :", MODEL)
print("벡터 차원   :", dim)
print("최대 입력   :", model.max_seq_length, "토큰")
print()

# ── (1) 길이가 다른 문장 3개 ─────────────────────────────────────
sentences = [
    "고양이",                                              # 아주 짧음
    "고양이가 소파 위에 있다",                              # 보통
    "제공된 자료에 따르면 출발 25일 전 취소 시 '출발 29일 전 ~ 20일 전' "
    "구간에 해당하여 요금의 10%인 89,000원이 취소 수수료로 부과됩니다.",  # 김
]

# encode() 가 문장을 벡터로 바꾼다. 리스트를 주면 한 번에 여러 개를 처리한다.
vectors = model.encode(sentences)

print("문장 길이가 달라도 벡터 크기는 같다")
print("-" * 62)
print("%-6s %-8s %s" % ("글자수", "벡터크기", "문장"))
for s, v in zip(sentences, vectors):
    preview = s if len(s) <= 28 else s[:28] + "..."
    print("%-6d %-8s %s" % (len(s), v.shape, preview))
print()
print("=> 글자 수와 무관하게 전부 (%d,) 이다." % dim)
print("   문장을 '의미 공간의 좌표 한 점' 으로 바꾸기 때문이다.")
print("   좌표의 축이 %d개인 것이고, 문장이 길다고 축이 늘지 않는다." % dim)
print()

# ── (2) 벡터 안을 들여다본다 ─────────────────────────────────────
v = vectors[1]
print("두 번째 문장 '%s' 의 벡터" % sentences[1])
print("-" * 62)
print("앞 8개 :", ", ".join("%+.4f" % x for x in v[:8]))
print("뒤 4개 :", ", ".join("%+.4f" % x for x in v[-4:]))
print("최솟값 %+.4f / 최댓값 %+.4f / 평균 %+.4f" % (v.min(), v.max(), v.mean()))
print()
print("=> 사람이 읽고 의미를 알 수 있는 숫자가 아니다. 이 숫자들 자체가 아니라")
print("   '두 벡터가 서로 얼마나 가까운가' 만 쓴다. 그게 STEP 3 의 코사인 유사도다.")
print()

# ── (3) 최대 입력 길이 한계 ──────────────────────────────────────
# 임베딩 모델은 생성 모델과 달리 입력 길이가 짧다. 이 모델은 128 토큰이다.
# 우리 policy.md 는 873 토큰이므로 통째로 넣으면 대부분이 잘린다.
tok = model.tokenizer
long_text = sentences[2]
n_tok = len(tok.encode(long_text))
print("입력 길이 한계")
print("-" * 62)
print("긴 문장 1개            %4d 토큰  (한계 %d 이내)" % (n_tok, model.max_seq_length))
print("data/policy.md 전문     873 토큰  <- 한계를 6.8배 초과")
print()
print("=> 문서를 통째로 임베딩할 수 없다. 조각내야 한다(chunking).")
print("   RAG 가 문서를 쪼개는 이유가 검색 때문만이 아니라 여기에도 있다.")
