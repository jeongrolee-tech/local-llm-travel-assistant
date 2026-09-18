import io, json

policy = io.open("data/policy.md", encoding="utf-8").read()
sysp = io.open("data/system_prompt.txt", encoding="utf-8").read().replace("{policy}", policy)
qs = json.load(io.open("data/questions.json", encoding="utf-8"))["questions"]
q = qs[0]

from ollama import Client

MODEL = "qwen3.5:9b"
QUESTION = "출발 25일 전에 취소하면 수수료가 얼마인가요?"
# 내 PC에서 실행 중인 Ollama에 연결합니다.
client = Client(host="http://127.0.0.1:11434", timeout=180)
print("Ollama에 질문을 보냈습니다. 답변을 기다려 주세요.")
response = client.chat(
    model=MODEL,
    # messages=[{"role": "user", "content": QUESTION}],
    messages=[{"role": "system", "content": sysp},
                {"role": "user", "content": q["question"]}],
    stream=False,
    # thinking 모드 스위치입니다. "3문장 이내" 제약은 여기가 아니라
    # data/system_prompt.txt 규칙 2에서 옵니다.
    # Qwen3.5 계열은 thinking이 기본 켜짐이라, num_ctx=4096에서 켜두면
    # thinking이 컨텍스트를 다 써서 빈 응답이 나옵니다 (실측 62.2초, 본문 없음).
    think=False,
    options={
            # ─────────────────────────────────────────────────────────────
            # temperature 를 0 으로 두지 않는 이유
            #
            # 발제문은 생성 설정의 "값"을 정해주지 않습니다. 동일하게 적용하고
            # 결과 파일에 기록하라고만 합니다. 따라서 값은 우리가 정합니다.
            #
            # 실측: 같은 질문을 2회 실행했을 때
            #   temperature = 0    -> 두 응답이 완전히 동일 (결정론적)
            #   temperature = 0.2  -> 두 응답이 갈림
            #
            # data/rubric.md 의 P4 판정 기준이 "Q9·Q10 4회 중 1점 이상 3회 이상"
            # 이고, 채점 주의사항도 "같은 질문의 2회 응답은 각각 채점한다" 입니다.
            # 둘 다 회차마다 답이 달라질 수 있다는 전제입니다. 루브릭은 첫 호출
            # 이전에 확정·커밋(7800589)했고 실험 중 수정하지 않으므로, 이와
            # 정합하려면 temperature > 0 이어야 했습니다.
            #
            # ── 다만 이 선택이 최선이었는지는 별개입니다 ──────────────────
            #
            # 정정: temperature=0 이어도 P4 는 작동합니다. 4회가 0/4·2/4·4/4
            # 중 하나로만 나오므로 "3회 이상" 이 "Q9·Q10 둘 다 통과" 라는 더
            # 엄격한 기준으로 바뀔 뿐, 성립하지 않는 것은 아닙니다.
            #
            # 실제 트레이드오프는 이렇습니다.
            #   재현 가능한 비교가 목적   -> temperature=0
            #   운영 안정성 측정이 목적   -> temperature>0 이되 n >= 10
            #   본 실험 (temp 0.2, n=2)  -> 그 중간. 변동은 생기는데 그 변동을
            #                                측정할 표본이 없습니다.
            #
            # 대가가 실측으로 확인됐습니다. 선택 실습 A에서 같은 태그·같은 설정을
            # 재실행했더니 종합 점수가 6.65점 움직였습니다 (docs/06_quantization.md
            # 7절). 질문 10개 x 2회 규모에서는 temperature=0 이 변별력에 유리했을
            # 것으로 판단합니다. 반복 2회는 발제문 지정값이라 늘릴 수 없었습니다.
            #
            # 그래도 결과를 본 뒤 설정을 바꾸지는 않았습니다. 사전 확정·사후 불변
            # 원칙을 지키고, 더 나았을 선택지는 README 4절·10절에 기록했습니다.
            #
            # seed 도 같은 이유로 고정하지 않습니다.
            # ─────────────────────────────────────────────────────────────
            "temperature": 0.2,   # 본 실험 고정값 (README 4절)
            "num_ctx": 4096,      # 입력 + 출력이 들어갈 전체 컨텍스트
            # "num_predict": 256, # 출력 한도. 본 실험은 미설정(모델 기본값)
            },
)

print("\n[Ollama 답변]")
print(response.message.content)
