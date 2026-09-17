"""후보 선별 예비 실행 (본 실험 아님).

모델 N개 × 질문 10개 × 1회. 로컬 A·B를 고르기 위한 선별용이며,
본 실험(최종 후보 2개 × 10문제 × 2회 = 40회) 집계에는 포함하지 않는다.

실행:
    uv run python docs/pilot/run_pilot.py qwen3.5:9b gemma4:e4b

출력은 이 스크립트와 같은 디렉터리에 pilot_<타임스탬프>.txt / .jsonl 로 저장된다.
"""

import io
import json
import os
import sys
import time
import urllib.request

API = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# 본 실험과 동일하게 고정한 생성 설정 (README 4절)
NUM_CTX = 4096
TEMPERATURE = 0.2
THINK = False

policy = io.open(os.path.join(ROOT, "data", "policy.md"), encoding="utf-8").read()
sysp = io.open(os.path.join(ROOT, "data", "system_prompt.txt"), encoding="utf-8").read()
sysp = sysp.replace("{policy}", policy)
QS = json.load(io.open(os.path.join(ROOT, "data", "questions.json"), encoding="utf-8"))["questions"]


def post(path, payload):
    req = urllib.request.Request(API + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def ps():
    return json.load(urllib.request.urlopen(API + "/api/ps", timeout=60))


def ask(model, question):
    t0 = time.time()
    r = post("/api/chat", {
        "model": model, "stream": False, "think": THINK,
        "options": {"num_ctx": NUM_CTX, "temperature": TEMPERATURE},
        "messages": [{"role": "system", "content": sysp},
                     {"role": "user", "content": question}],
    })
    return time.time() - t0, r


def main(models):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    txt = io.open(os.path.join(HERE, "pilot_%s.txt" % stamp), "w", encoding="utf-8")
    rows = io.open(os.path.join(HERE, "pilot_%s.jsonl" % stamp), "w", encoding="utf-8")
    txt.write("후보 선별 예비 실행 (본 실험 집계 제외)\n")
    txt.write("설정: num_ctx=%d, temperature=%s, think=%s, 대화이력 없음, 각 1회\n\n"
              % (NUM_CTX, TEMPERATURE, THINK))

    for model in models:
        post("/api/generate", {"model": model, "keep_alive": 0})
        time.sleep(1)
        warm_el, warm_r = ask(model, "안녕하세요")   # 워밍업 1회 — 집계 분리
        txt.write("#" * 74 + "\n### %s\n" % model)
        txt.write("워밍업 1회: %.2fs (load %.2fs) — 집계 제외\n\n"
                  % (warm_el, warm_r.get("load_duration", 0) / 1e9))

        total_t, total_out, fails = 0.0, 0, 0
        for q in QS:
            try:
                el, r = ask(model, q["question"])
            except Exception as exc:                      # 호출 실패는 별도 기록
                txt.write("[%s] 호출 실패: %s\n\n" % (q["id"], exc))
                rows.write(json.dumps({"model": model, "qid": q["id"], "error": str(exc)},
                                      ensure_ascii=False) + "\n")
                fails += 1
                continue
            msg = r.get("message", {})
            answer = (msg.get("content") or "").strip()
            pe = r.get("prompt_eval_count", 0)
            ec = r.get("eval_count", 0)
            ed = r.get("eval_duration", 0)
            total_t += el
            total_out += ec
            if not answer:
                fails += 1
            txt.write("[%s] %s / grounded=%s\n" % (q["id"], q["type"], q.get("grounded")))
            txt.write("  Q    : %s\n" % q["question"])
            txt.write("  기대 : %s\n" % q["expected"])
            txt.write("  답변 : %s\n" % (answer if answer else "(빈 응답)"))
            # eval_duration <= 0 이면 tok/s를 계산하지 않고 사유를 남긴다 (0으로 채우지 않음)
            txt.write("  측정 : %.2fs | in %d + out %d tok | %s\n\n" % (
                el, pe, ec,
                ("%.1f tok/s" % (ec / (ed / 1e9))) if ed > 0 else "tok/s 계산 불가 (eval_duration<=0)"))
            rows.write(json.dumps({
                "model": model, "qid": q["id"], "type": q["type"],
                "grounded": q.get("grounded"), "elapsed_s": round(el, 3),
                "prompt_tok": pe, "out_tok": ec,
                "gen_tok_per_s": round(ec / (ed / 1e9), 2) if ed > 0 else None,
                "answer": answer,
            }, ensure_ascii=False) + "\n")

        ok = len(QS) - fails
        vram = total = 0
        for x in ps().get("models", []):
            if model in (x.get("name"), x.get("model")):
                vram, total = x.get("size_vram", 0), x.get("size", 0)
        txt.write("합계: %d/%d 응답 반환 | 총 %.1fs (평균 %.2fs) | 출력 %d tok | "
                  "VRAM %.0f MiB / %.0f MiB\n\n"
                  % (ok, len(QS), total_t, total_t / max(ok, 1), total_out,
                     vram / 1048576, total / 1048576))
        txt.flush()
        post("/api/generate", {"model": model, "keep_alive": 0})

    txt.close()
    rows.close()
    print("saved to", HERE)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("사용법: python docs/pilot/run_pilot.py <모델 태그> [모델 태그 ...]")
    main(sys.argv[1:])
