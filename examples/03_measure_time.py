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

load_s = response.load_duration / 1_000_000_000
print(f"\n전체 응답 시간: {elapsed:.2f}초")
print(f"  └ 모델 로딩 : {load_s:.2f}초")
print(f"  └ 나머지    : {elapsed - load_s:.2f}초  (프롬프트 처리 + 생성)")
print(f"출력 토큰: {response.eval_count}")

if load_s < 0.1:
    print("\n→ 로딩 시간이 0에 가깝습니다. 모델이 이미 GPU에 올라가 있었습니다.")
    print("  워밍업 효과를 보려면 'ollama stop qwen3.5:9b' 로 내린 뒤 다시 실행하세요.")
else:
    print(f"\n→ 이번 실행은 모델 로딩에 {load_s:.2f}초를 썼습니다.")
    print("  본 실험에서 워밍업 1회를 집계에서 분리하는 이유가 이것입니다.")

# 전체 응답 시간은 첫 토큰 시간(TTFT)이 아닙니다. 별도로 측정하지 않았으므로
# TTFT 로 표기하지 않습니다. (발제문 주의사항)
# 출력 토큰 수가 회차마다 다르면 전체 응답 시간도 달라집니다. 시간만 보고
# 모델 우열을 판단하지 마십시오.
