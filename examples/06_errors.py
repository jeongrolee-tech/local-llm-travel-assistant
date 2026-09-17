"""STEP 8 — 호출 실패를 일부러 일으켜 예외 유형과 메시지를 확인한다.

발제문 STEP 4: "오류 증상과 확인 내용을 기록합니다."
발제문 Quality 원칙: "호출 실패는 품질 점수와 별도로 기록하고 성공 응답으로
대체하지 않습니다."

본 실험 스크립트가 어떤 예외를 잡아야 40회가 중간에 멈추지 않는지 확인하는 것이
목적이다.

실행:
    uv run python examples/06_errors.py
"""

import httpx
from ollama import Client, ResponseError

HOST = "http://127.0.0.1:11434"
MODEL = "qwen3.5:9b"

client = Client(host=HOST, timeout=180)


def probe(label, fn):
    try:
        fn()
        print("  %-22s -> 예외 없음 (호출 성공)" % label)
        return None
    except Exception as e:
        print("  %-22s -> %s" % (label, type(e).__name__))
        print("  %-22s    %s" % ("", str(e)[:100]))
        return e


print("호출 실패 유형 확인\n")

# 1. 존재하지 않는 태그 — 형식은 맞지만 없는 모델
probe("없는 태그", lambda: client.chat(
    model="qwen3.5:999b",
    messages=[{"role": "user", "content": "안녕"}], stream=False))

# 2. 형식이 잘못된 이름
probe("잘못된 이름 형식", lambda: client.chat(
    model="존재하지-않는-모델:99b",
    messages=[{"role": "user", "content": "안녕"}], stream=False))

# 3. 타임아웃 — 본 실험에서 실제로 날 수 있는 실패
slow = Client(host=HOST, timeout=0.001)
probe("타임아웃", lambda: slow.chat(
    model=MODEL, messages=[{"role": "user", "content": "안녕"}], stream=False))

# 4. 서버 연결 불가 — Ollama 가 꺼져 있는 경우
dead = Client(host="http://127.0.0.1:19999", timeout=3)
probe("서버 연결 불가", lambda: dead.chat(
    model=MODEL, messages=[{"role": "user", "content": "안녕"}], stream=False))

print()
print("본 실험 스크립트가 잡아야 할 예외 — 세 부류가 서로 다른 계통이다")
print("  ollama.ResponseError   모델 없음(404), 이름 형식 오류(400) 등 서버 응답 오류")
print("  httpx.ReadTimeout      응답이 timeout 안에 오지 않음")
print("  ConnectionError        Ollama 서버에 연결 불가 (파이썬 내장, OSError 하위)")

print("\n상속 관계 — 하나의 상위 클래스로 묶이지 않는다")
print("  ollama.ResponseError  <-", " <- ".join(c.__name__ for c in ResponseError.__mro__[1:3]))
print("  httpx.ReadTimeout     <-", " <- ".join(c.__name__ for c in httpx.ReadTimeout.__mro__[1:4]))
print("  ConnectionError       <-", " <- ".join(c.__name__ for c in ConnectionError.__mro__[1:4]))
print()
print("  httpx.HTTPError 로 ollama.ResponseError 가 잡히는가:",
      issubclass(ResponseError, httpx.HTTPError))
print("  httpx.HTTPError 로 ConnectionError 가 잡히는가    :",
      issubclass(ConnectionError, httpx.HTTPError))

print()
print("→ 좁게 잡으면 빠뜨립니다. 본 실험은 호출 단위로 넓게 잡고 예외 유형을 기록합니다.")
print()
print("      try:")
print("          r = client.chat(...)")
print("      except Exception as e:                       # 40회가 중간에 멈추면 안 된다")
print("          rec = {'status': 'error',")
print("                 'error_type': type(e).__name__,   # ResponseError / ReadTimeout / ...")
print("                 'error_msg': str(e)}")
print("          errors.write(json.dumps(rec) + '\\n')     # results/errors.jsonl")
print()
print("호출 실패는 품질 점수와 분리해 기록하고 성공 응답으로 대체하지 않는다.")
print("성공 수 / 전체 시도 수를 모델별로 따로 표시한다. (발제문 Quality 원칙)")
