import json
import time
import urllib.request

API = "http://127.0.0.1:11434"
RAW = "hf.co/mradermacher/kanana-2-3b-instruct-GGUF:Q4_K_M"
FIX = "kanana2-3b-chatml:q4km"
REF = "qwen3.5:4b"   # 렌더러가 있는 공식 배포본 (대조군)


def post(path, payload):
    req = urllib.request.Request(API + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def probe(model, label, messages):
    post("/api/generate", {"model": model, "keep_alive": 0})
    time.sleep(0.5)
    r = post("/api/chat", {
        "model": model, "stream": False, "think": False,
        "options": {"num_ctx": 4096, "temperature": 0, "seed": 42},
        "messages": messages,
    })
    pe = r.get("prompt_eval_count")
    ans = (r.get("message", {}).get("content") or "").strip()
    print("  %-14s prompt_eval_count=%-5s  답변=%r" % (label, pe, ans[:70]))
    post("/api/generate", {"model": model, "keep_alive": 0})
    return pe, ans


print("=== A. 짧은 단일 user 메시지 ===")
m1 = [{"role": "user", "content": "안녕"}]
for mdl, lab in [(RAW, "raw"), (FIX, "chatml"), (REF, "qwen3.5:4b")]:
    probe(mdl, lab, m1)

print("\n=== B. system + user (역할 분리) ===")
m2 = [{"role": "system", "content": "너는 여행사 상담원이다."},
      {"role": "user", "content": "안녕"}]
for mdl, lab in [(RAW, "raw"), (FIX, "chatml"), (REF, "qwen3.5:4b")]:
    probe(mdl, lab, m2)

print("\n=== C. /api/generate (raw prompt, 템플릿 미적용 경로) ===")
for mdl, lab in [(RAW, "raw"), (FIX, "chatml")]:
    post("/api/generate", {"model": mdl, "keep_alive": 0})
    time.sleep(0.5)
    r = post("/api/generate", {"model": mdl, "prompt": "안녕", "stream": False, "raw": True,
                               "options": {"num_ctx": 4096, "temperature": 0, "seed": 42,
                                           "num_predict": 16}})
    print("  %-14s prompt_eval_count=%-5s  답변=%r"
          % (lab, r.get("prompt_eval_count"), (r.get("response") or "").strip()[:60]))
    post("/api/generate", {"model": mdl, "keep_alive": 0})
