from time import perf_counter
from ollama import Client

MODEL = "qwen3.5:9b"
client = Client(host="http://127.0.0.1:11434", timeout=180)

start = perf_counter()
r = client.chat(
    model=MODEL,
    messages=[{"role": "user", "content": "안녕하세요"}],
    stream=False, think=False,
    # temperature 를 0 으로 두지 않는 이유는 01_ollama_chat.py 주석과
    # README 4절 참조. 요약: rubric.md 의 P4 판정이 "4회 중 3회 이상" 이라
    # 회차마다 답이 달라져야 성립하는데, temperature=0 이면 반복이 전부 같습니다.
    options={"temperature": 0.2, "num_ctx": 4096},   # 본 실험 고정값
)
elapsed = perf_counter() - start

print("전체 응답 시간 :", round(elapsed, 2), "초")
print("로딩 시간      :", round(r.load_duration / 1e9, 2), "초")
print("입력 토큰      :", r.prompt_eval_count)
print("출력 토큰      :", r.eval_count)
if r.eval_duration and r.eval_duration > 0:
    print("생성 속도      :", round(r.eval_count / (r.eval_duration / 1e9), 1), "tok/s")
else:
    print("생성 속도      : 계산 불가 (eval_duration <= 0)")
print("종료 사유      :", r.done_reason)
