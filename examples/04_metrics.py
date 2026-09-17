import io, json

policy = io.open("data/policy.md", encoding="utf-8").read()
sysp = io.open("data/system_prompt.txt", encoding="utf-8").read().replace("{policy}", policy)
qs = json.load(io.open("data/questions.json", encoding="utf-8"))["questions"]
q = qs[0]


from time import perf_counter
from ollama import Client

MODEL = "qwen3.5:9b"
client = Client(host="http://127.0.0.1:11434", timeout=180)

start = perf_counter()
r = client.chat(
    model=MODEL,
    # messages=[{"role": "user", "content": "안녕하세요"}],
    messages=[{"role": "system", "content": sysp},
            {"role": "user", "content": q["question"]}],
    stream=False, think=False,
    # temperature 를 0 으로 두지 않는 이유는 01_ollama_chat.py 주석과
    # README 4절 참조. 요약: rubric.md 의 P4 판정이 "4회 중 3회 이상" 이라
    # 회차마다 답이 달라져야 성립하는데, temperature=0 이면 반복이 전부 같습니다.
    # num_ctx 를 8192 로 바꾸면 qwen3.5:9b 가 CPU 로 분할됩니다.
    # 100% GPU -> 88% GPU, 생성 속도 59.8 -> 53.0 tok/s (docs/walkthrough_log.md STEP 5)
    options={"temperature": 0.2, "num_ctx": 4096},   # 본 실험 고정값. 8192 로 바꾸면 CPU 분할됨. 100% GPU -> 88% GPU, 생성 속도 59.8 -> 53.0 tok/s (docs/walkthrough_log.md STEP 5)    
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

for m in client.ps().models:
    if m.model == MODEL:
        vram, total = m.size_vram, m.size
        print("VRAM (GPU 적재):", round(vram / 1048576), "MiB")
        print("전체 적재      :", round(total / 1048576), "MiB")
        print("적재 상태      :", "100% GPU" if vram == total
              else "%.0f%% GPU / CPU 분할" % (vram / total * 100))
        print("digest         :", m.digest[:12])
        print("quantization   :", m.details.quantization_level)
        print("실제 context   :", m.context_length)
