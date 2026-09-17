from ollama import Client

MODEL = "qwen3.5:9b"
QUESTION = "출발 25일 전에 취소하면 수수료가 얼마인가요?"
# 내 PC에서 실행 중인 Ollama에 연결합니다.
client = Client(host="http://127.0.0.1:11434", timeout=180)
print("Ollama에 질문을 보냈습니다. 답변을 기다려 주세요.")
response = client.chat(
    model=MODEL,
    messages=[{"role": "user", "content": QUESTION}],
    stream=False,
    think=False, # 3문장 제한
    options={
            "temperature": 0,
            "num_ctx": 4096,      # 입력 + 출력이 들어갈 전체 컨텍스트
            # "num_predict": 256,   # 생성할 최대 토큰
            },
)

print("\n[Ollama 답변]")
print(response.message.content)
