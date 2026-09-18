# -*- coding: utf-8 -*-
"""선택 실습 A — Quantization 비교 집계.

    uv run python src/aggregate_quant.py

채점은 data/rubric.md 기준이며, 판단 규칙 3가지는 aggregate.py 와 완전히 같다.
(1) 루브릭 문언 우선 — 금액 미제시는 오류가 아니라 누락이므로 항목2에서 감점
(2) 하나의 결함은 한 항목에서만 감점
(3) 항목3은 "3문장 이내" 와 "근거 항목명 명시" 두 가지만 본다

비교 대상은 같은 모델(gemma4 E4B-it)의 두 양자화본이다. 발제문의
"서로 다른 모델의 차이를 양자화 효과로 해석하지 않는다" 를 지키기 위해
다른 모델은 이 집계에 넣지 않는다.

출력
    results/quant_scores.csv    회차별 항목 점수와 근거
    results/quant_summary.csv   양자화별 요약
"""
import csv
import io
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

Q4 = "gemma4:e4b"              # Q4_K_M
Q8 = "gemma4:e4b-it-q8_0"      # Q8_0

# (model, qid, run): (항목1, 항목2, 항목3, 항목4, 항목5|None, 근거)
S = {
 # ── Q4_K_M (4bit) ────────────────────────────────────────────────
 (Q4,"Q1",1): (2,1,2,2,None,"'출발 29일 전 ~ 20일 전' 구간·10% 정확, 항목명 인용. 89,000원 미제시(항목2)"),
 (Q4,"Q1",2): (2,1,2,2,None,"동일"),
 (Q4,"Q2",1): (2,1,1,2,None,"30% 구간 정확. 890,000원이 문서에 있는데 '계산하기 어렵다'며 534,000원 미제시(항목2). 4문장으로 3문장 초과(항목3)"),
 (Q4,"Q2",2): (2,1,1,2,None,"동일"),
 (Q4,"Q3",1): (2,1,2,2,None,"소아 구간·80%·890,000원 정확. 712,000원 미계산(항목2)"),
 (Q4,"Q3",2): (0,0,1,2,None,"문서 2절에 소아 구간이 있는데 '만 3세 아동 별도 요금 정보 확인되지 않음'이라 답변 — 문서와 배치(항목1 0점). 무관한 유아 150,000원만 제시해 핵심 조건 대부분 누락(항목2 0점). 근거 항목명이 질문과 무관(항목3)"),
 (Q4,"Q4",1): (2,1,1,2,None,"6개월 조건·미달 판단 정확. 여권 갱신 안내 누락(항목2). '제공된 자료에 따르면'만 있고 항목명 없음(항목3)"),
 (Q4,"Q4",2): (2,1,1,2,None,"'확인이 어렵습니다'로 판단 회피 후 조건만 인용, 갱신 안내 누락(항목2). 항목명 미제시(항목3)"),
 (Q4,"Q5",1): (2,2,2,2,None,"2인 1실 기준과 220,000원 모두 제시, '요금 기준' 항목명 명시"),
 (Q4,"Q5",2): (2,2,2,2,None,"'2. 요금 기준' 항목명과 220,000원 제시, 3문장 유지"),
 (Q4,"Q6",1): (2,1,2,2,None,"석식 불포함 결론 명확, '불포함 사항' 항목명 명시. 조식 3회 포함 안내 누락(항목2)"),
 (Q4,"Q6",2): (2,1,2,2,None,"동일. 선택관광·팁 등 무관 정보가 붙었으나 사실과 일치"),
 (Q4,"Q7",1): (2,2,1,2,None,"면제·증빙·개인사정 구분 모두 정확. 근거 항목명 미제시(항목3)"),
 (Q4,"Q7",2): (2,2,1,2,None,"동일"),
 (Q4,"Q8",1): (2,1,2,2,None,"인용한 1인 객실 추가요금 220,000원은 문서와 일치→항목1 2점. 취소 수수료 적용 미언급(항목2)"),
 (Q4,"Q8",2): (2,1,2,2,None,"취소 수수료 산정 기준 제시 후 추가 정보 요청. 1인 객실 추가요금 영향 미언급(항목2)"),
 (Q4,"Q9",1): (2,2,2,2,2,"확인 불가 명시 + 담당자 안내, 보장 한도를 지어내지 않음"),
 (Q4,"Q9",2): (2,2,2,2,2,"동일"),
 (Q4,"Q10",1):(2,2,2,2,2,"범위 밖 명시 + 담당자 안내, 추측 없음"),
 (Q4,"Q10",2):(2,2,2,2,2,"동일"),

 # ── Q8_0 (8bit) ──────────────────────────────────────────────────
 (Q8,"Q1",1): (2,1,2,2,None,"구간·10% 정확, '취소 수수료 규정' 항목명 명시. 89,000원 미제시(항목2)"),
 (Q8,"Q1",2): (2,1,2,2,None,"동일"),
 (Q8,"Q2",1): (2,2,2,2,None,"30% 구간·890,000원·2인 1,780,000원·534,000원까지 계산 완결. '상품 개요' 항목명 명시"),
 (Q8,"Q2",2): (2,2,2,2,None,"동일. '취소 수수료 규정'·'상품 개요' 두 항목명 인용"),
 (Q8,"Q3",1): (2,1,2,2,None,"소아 구간·80%·890,000원 정확. 712,000원 미계산(항목2)"),
 (Q8,"Q3",2): (2,2,2,2,None,"소아 80%·712,000원까지 계산 완결, '상품 개요' 항목명 명시"),
 (Q8,"Q4",1): (2,1,1,2,None,"6개월 조건·미달 판단 정확. 여권 갱신 안내 누락(항목2). 항목명 미제시(항목3)"),
 (Q8,"Q4",2): (2,1,1,2,None,"동일"),
 (Q8,"Q5",1): (2,2,2,2,None,"2인 1실 기준과 220,000원 모두 제시, '요금 기준' 항목명 명시"),
 (Q8,"Q5",2): (2,2,2,2,None,"동일"),
 (Q8,"Q6",1): (2,1,2,2,None,"석식 불포함 명확, '4. 불포함 사항' 항목명 명시. 조식 3회 포함 안내 누락(항목2)"),
 (Q8,"Q6",2): (2,1,2,2,None,"동일. '[4. 불포함 사항]' 대괄호 표기이나 그대로 발송 가능한 수준이라 항목4 감점 없음"),
 (Q8,"Q7",1): (2,1,1,2,None,"면제·증빙 정확. 개인 사정 미탑승은 적용 대상이 아니라는 구분 누락(항목2). 항목명 미제시(항목3)"),
 (Q8,"Q7",2): (2,1,1,2,None,"동일"),
 (Q8,"Q8",1): (1,1,2,2,None,"'출발일 기준 19일 전부터의 변경은 취소 후 재예약'은 문서 6절에 실재하나 6절은 일정(출발일) 변경 규정이며 인원 변경에는 적용되지 않음 — 인접 규정 혼동(항목1). 결론인 제5항 적용은 맞음. 1인 객실 추가요금 미언급(항목2)"),
 (Q8,"Q8",2): (2,1,2,2,None,"인원 변경으로 분류하고 취소 수수료 산정 기준 제시, 단정하지 않음. 1인 객실 추가요금 미언급(항목2)"),
 (Q8,"Q9",1): (2,2,2,1,2,"확인 불가 명시 + 담당자 안내. '확인해 주시기를 안내해 드립니다'가 비문(항목4)"),
 (Q8,"Q9",2): (2,2,2,2,2,"확인 불가 명시 + 담당자 안내, 지어내지 않음"),
 (Q8,"Q10",1):(2,2,2,2,2,"범위 밖 명시 + 담당자 안내, 추측 없음"),
 (Q8,"Q10",2):(2,2,2,2,2,"동일"),
}

rows = [json.loads(l) for l in io.open("results/quant_raw.jsonl", encoding="utf-8")]
idx = {(r["model"], r["qid"], r["run"]): r for r in rows}
assert len(idx) == 40 and set(idx) == set(S), "양자화 비교 레코드와 채점표가 맞지 않음"


def total_and_max(v):
    """Q9·Q10 은 항목5까지 10점 만점, 나머지는 8점 만점."""
    items = [x for x in v[:5] if x is not None]
    return sum(items), len(items) * 2


# ── 회차별 점수 ──────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
with io.open("results/quant_scores.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "quantization", "qid", "qtype", "run",
                "item1", "item2", "item3", "item4", "item5",
                "total", "max", "normalized_100",
                "elapsed_s", "out_tok", "gen_tok_per_s", "vram_mib", "processor",
                "score_evidence"])
    for key in sorted(S, key=lambda k: (k[0], int(k[1][1:]), k[2])):
        v = S[key]
        r = idx[key]
        tot, mx = total_and_max(v)
        w.writerow([key[0], r["quantization"], key[1], r["qtype"], key[2],
                    v[0], v[1], v[2], v[3], "" if v[4] is None else v[4],
                    tot, mx, round(tot / mx * 100, 1),
                    r["elapsed_s"], r["out_tok"], r["gen_tok_per_s"],
                    r["vram_mib"], r["processor"], v[5]])

# ── 양자화별 요약 ────────────────────────────────────────────────
summary = []
for model in (Q4, Q8):
    keys = [k for k in S if k[0] == model]
    norm = [total_and_max(S[k])[0] / total_and_max(S[k])[1] * 100 for k in keys]
    recs = [idx[k] for k in keys]
    items = {}
    for i in range(4):
        items["item%d" % (i + 1)] = round(st.mean([S[k][i] for k in keys]), 2)
    k5 = [k for k in keys if S[k][4] is not None]
    items["item5"] = round(st.mean([S[k][4] for k in k5]), 2)
    bytype = {}
    for t in sorted({r["qtype"] for r in recs}):
        sel = [k for k in keys if idx[k]["qtype"] == t]
        bytype[t] = round(st.mean([total_and_max(S[k])[0] / total_and_max(S[k])[1] * 100
                                   for k in sel]), 1)
    summary.append({
        "model": model,
        "quantization": recs[0]["quantization"],
        "n": len(keys),
        "score_100": round(st.mean(norm), 2),
        "vram_mib": recs[0]["vram_mib"],
        "processor": recs[0]["processor"],
        "gen_tok_per_s": round(st.mean([r["gen_tok_per_s"] for r in recs]), 1),
        "elapsed_s_mean": round(st.mean([r["elapsed_s"] for r in recs]), 3),
        "out_tok_mean": round(st.mean([r["out_tok"] for r in recs]), 1),
        "prompt_tok": recs[0]["prompt_tok"],
        "success": "%d/%d" % (len(recs), len(keys)),
        **items, **{"type_" + t: v for t, v in bytype.items()},
    })

with io.open("results/quant_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)

# ── 콘솔 출력 ────────────────────────────────────────────────────
a, b = summary
print("선택 실습 A — Quantization 비교  (같은 모델 gemma4 E4B-it, 양자화만 다름)")
print()
print("%-16s %-8s %-8s %-9s %-11s %-9s" % ("", "4bit", "8bit", "차이", "항목", "판정"))
print("-" * 66)


def line(label, x, y, fmt="%.2f", better="high"):
    d = y - x
    if abs(d) < 1e-9:
        verdict = "동일"
    elif (d > 0) == (better == "high"):
        verdict = "8bit 우세"
    else:
        verdict = "4bit 우세"
    print("%-16s %-8s %-8s %-9s %s" % (label, fmt % x, fmt % y, ("%+.2f" % d), verdict))


line("종합 점수(100)", a["score_100"], b["score_100"])
line("VRAM (MiB)", a["vram_mib"], b["vram_mib"], "%.0f", "low")
line("생성속도 tok/s", a["gen_tok_per_s"], b["gen_tok_per_s"], "%.1f")
line("응답시간 평균s", a["elapsed_s_mean"], b["elapsed_s_mean"], "%.3f", "low")
line("한국어 표현", a["item4"], b["item4"])
print()
print("항목별 (0~2점)")
for i in range(1, 6):
    k = "item%d" % i
    print("  항목%d  4bit %.2f | 8bit %.2f" % (i, a[k], b[k]))
print()
print("문항 유형별 (100점 환산)")
for t in ["normal", "boundary", "insufficient_info", "out_of_scope"]:
    ka, kb = "type_" + t, "type_" + t
    if ka in a:
        print("  %-18s 4bit %5.1f | 8bit %5.1f" % (t, a[ka], b[kb]))
print()

# ── 재현성 점검: 이번 4bit 재실행 vs 필수 실험 4bit ──────────────
if os.path.exists("results/local_raw.jsonl"):
    main = [json.loads(l) for l in io.open("results/local_raw.jsonl", encoding="utf-8")
            if json.loads(l)["model"] == Q4]
    if main:
        print("재현성 점검 — 같은 태그·같은 설정을 다른 세션에서 재실행한 값")
        print("  생성속도  필수 실험 %.1f tok/s | 이번 재실행 %.1f tok/s (%+.1f%%)"
              % (st.mean([r["gen_tok_per_s"] for r in main]), a["gen_tok_per_s"],
                 (a["gen_tok_per_s"] / st.mean([r["gen_tok_per_s"] for r in main]) - 1) * 100))
        print("  VRAM      필수 실험 %s MiB | 이번 재실행 %s MiB"
              % (main[0]["vram_mib"], a["vram_mib"]))
        print("  입력 토큰 필수 실험 %d | 이번 재실행 %d"
              % (main[0]["prompt_tok"], a["prompt_tok"]))
print()
print("results/quant_scores.csv, results/quant_summary.csv 생성")
