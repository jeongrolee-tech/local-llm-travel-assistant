from time import perf_counter

from ollama import Client

MODEL = "qwen3.5:9b"
QUESTION = "출발 25일 전에 취소하면 수수료가 얼마인가요?"
client = Client(host="http://127.0.0.1:11434", timeout=180)

print("Ollama의 답변을 끝까지 받는 데 걸린 시간을 측정합니다.")
start = perf_counter()
response = client.chat(
    model=MODEL,
    messages=[{"role": "user", "content": QUESTION}],
    stream=False,
    think=False,
    # temperature 를 0 으로 두지 않는 이유는 01_ollama_chat.py 주석과
    # README 4절 참조. 요약: rubric.md 의 P4 판정이 "4회 중 3회 이상" 이라
    # 회차마다 답이 달라져야 성립하는데, temperature=0 이면 반복이 전부 같습니다.
    options={"temperature": 0.2, "num_ctx": 4096},   # 본 실험 고정값
)
elapsed = perf_counter() - start

print("\n[Ollama 답변]")
print(response.message.content)
print(f"\n전체 응답 시간: {elapsed:.2f}초")
# 첫 토큰 시간(TTFT)이 아닙니다. 필요하면 모델 로딩 시간도 포함됩니다.
