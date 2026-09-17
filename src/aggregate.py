# -*- coding: utf-8 -*-
"""품질 채점 결과를 results/scores.csv, results/summary.csv 로 집계한다.

    uv run python src/aggregate.py

채점은 data/rubric.md 기준이다. 아래 S 딕셔너리가 회차별 항목 점수와 근거이며,
응답 전문을 읽고 부여한 뒤 검토·수정한 값이다.

── 채점 판단 규칙 (전 회차 동일 적용) ─────────────────────────────

1. 루브릭 문언을 우선한다.
   항목1의 1점 기준은 "금액 계산 오류 또는 인접 구간 혼동" 이다. 금액을
   제시하지 않은 것은 오류가 아니라 누락이므로 항목1이 아니라 항목2에서
   감점한다.

2. 하나의 결함은 한 항목에서만 감점한다 (중복 감점 금지).
   항목들이 서로 다른 축을 재도록 설계되어 있다. 예: "문서에 답이 있는데
   확인이 어렵다고 회피" 한 경우, 진술 자체가 문서와 배치되지는 않으므로
   항목1은 2점을 유지하고 결론 누락을 항목2에서만 깎는다.

3. 채점 축에 따라 항목을 배정한다.
   항목3은 "3문장 이내" 와 "근거 항목명 명시" 두 가지만 본다. 마크다운 링크
   노출처럼 그대로 발송할 수 없는 문장은 항목4("수정 없이 고객에게 보낼 수
   있는 문장")에서 감점한다.

만점은 Q1~Q8 이 8점(항목1~4), Q9·Q10 이 10점(항목1~5)이다. 만점이 다르므로
합산 평균이 아니라 100점 환산으로 집계한다.
"""
import csv
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

Q = "qwen3.5:9b"
G = "gemma4:e4b"

# (model, qid, run): (항목1, 항목2, 항목3, 항목4, 항목5|None, 근거)
S = {
 (Q,"Q1",1): (2,1,1,2,None,"구간·10% 정확. 89,000원 미제시(항목2). '제공된 자료에 따르면'만 있고 규정 항목명 없음(항목3)"),
 (Q,"Q1",2): (2,1,2,2,None,"'출발 29일 전 ~ 20일 전' 항목명 인용. 89,000원 미제시"),
 (Q,"Q2",1): (2,2,2,1,None,"534,000원 정확. '([요금 기준](#2.-요금-기준))' 마크다운 링크 노출로 그대로 발송 불가(항목4)"),
 (Q,"Q2",2): (2,1,2,1,None,"534,000원 정확. 구간명 미언급, 끝에 '달라질 수 있습니다'로 흐림(항목2). 링크 노출(항목4)"),
 (Q,"Q3",1): (2,2,2,2,None,"소아 구간 분류·712,000원·계산식까지 정확"),
 (Q,"Q3",2): (2,2,2,2,None,"소아 80%·712,000원 정확"),
 (Q,"Q4",1): (2,1,2,2,None,"진술 자체는 문서와 일치(조치·예외는 실제로 문서에 없음)→항목1 2점. '충족하지 못한다' 판단과 갱신 안내 누락은 항목2에서만 감점"),
 (Q,"Q4",2): (2,1,2,0,None,"동일. '이라는项规定가 있으나' — 중국어 혼입으로 항목4 0점"),
 (Q,"Q5",1): (2,1,2,2,None,"220,000원 정확. 2인 1실 기준 미언급(항목2)"),
 (Q,"Q5",2): (2,2,2,2,None,"2인 1실 기준과 220,000원 모두 제시"),
 (Q,"Q6",1): (2,1,2,2,None,"인용한 내용은 문서와 일치→항목1 2점. 석식 불포함 결론 미제시는 항목2에서만 감점"),
 (Q,"Q6",2): (2,1,2,1,None,"결론 미제시는 항목2에서만 감점. '정확한 내용을 담당자 확인을 부탁드립니다' 비문(항목4)"),
 (Q,"Q7",1): (2,2,1,2,None,"면제·증빙·개인사정 구분 모두 정확. 근거 항목명 미제시(항목3)"),
 (Q,"Q7",2): (2,1,1,2,None,"면제·증빙 정확. 개인 사정 구분 누락(항목2). 항목명 미제시"),
 (Q,"Q8",1): (2,1,2,2,None,"단정하지 않음. 취소 수수료 부과 안내가 '규정은 있으나'로 약함(항목2)"),
 (Q,"Q8",2): (2,2,2,2,None,"객실 추가요금·취소 수수료·문서 6항 무료 변경까지 제시"),
 (Q,"Q9",1): (2,2,2,2,2,"보장 한도를 지어내지 않고 확인 불가 명시 + 담당자 안내"),
 (Q,"Q9",2): (2,2,2,2,2,"동일"),
 (Q,"Q10",1):(2,2,2,1,2,"범위 밖 명시. '해당 내용은 오사카 ... 약관에 명시되어 있습니다' 가 베트남 내용이 있다는 오독을 유발(항목4)"),
 (Q,"Q10",2):(2,2,2,1,2,"'약관에만 포함되어 있습니다' 로 다소 개선됐으나 같은 혼동 여지"),

 (G,"Q1",1): (2,1,2,2,None,"구간·10% 정확, 항목명 인용. 89,000원 미제시(항목2)"),
 (G,"Q1",2): (2,1,2,2,None,"동일"),
 (G,"Q2",1): (2,2,2,2,None,"30% 구간·1,780,000원·534,000원까지 계산 완결"),
 (G,"Q2",2): (2,2,2,2,None,"동일"),
 (G,"Q3",1): (2,1,2,1,None,"소아 구간 정확. 712,000원 미계산(항목2). '80%를 계산해야 합니다'로 고객에게 계산을 떠넘김(항목4)"),
 (G,"Q3",2): (2,1,2,2,None,"소아 80% 정확. 712,000원 미계산. 질문과 무관한 유아 요금 언급"),
 (G,"Q4",1): (2,1,2,2,None,"진술 자체는 문서와 일치→항목1 2점. 판단·갱신 안내 누락은 항목2에서만 감점"),
 (G,"Q4",2): (2,1,2,2,None,"동일"),
 (G,"Q5",1): (2,2,2,2,None,"2인 1실 기준과 220,000원 모두 제시"),
 (G,"Q5",2): (2,2,2,2,None,"동일"),
 (G,"Q6",1): (2,1,2,2,None,"석식 불포함 명확. 조식 3회 포함 안내 누락(항목2)"),
 (G,"Q6",2): (2,1,2,2,None,"동일"),
 (G,"Q7",1): (2,2,1,2,None,"면제·증빙·개인사정 구분 모두 정확. 근거 항목명 미제시(항목3)"),
 (G,"Q7",2): (2,2,1,2,None,"동일"),
 (G,"Q8",1): (2,1,2,2,None,"단정하지 않음. 1인 객실 추가요금 영향 미언급(항목2)"),
 (G,"Q8",2): (2,2,2,2,None,"1인 객실 추가요금 220,000원과 취소 수수료 적용 모두 제시"),
 (G,"Q9",1): (2,2,2,2,2,"확인 불가 명시 + 담당자 안내, 지어내지 않음"),
 (G,"Q9",2): (2,2,2,2,2,"동일"),
 (G,"Q10",1):(2,2,2,2,2,"범위 밖 명시 + 담당자 안내, 추측 없음"),
 (G,"Q10",2):(2,2,2,2,2,"동일"),
}

rows = [json.loads(l) for l in io.open("results/local_raw.jsonl", encoding="utf-8")]
idx = {(r["model"], r["qid"], r["run"]): r for r in rows}
assert len(idx) == 40 and set(idx) == set(S), "레코드와 채점표가 맞지 않음"

os.makedirs("results", exist_ok=True)
with io.open("results/scores.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "qid", "qtype", "run",
                "item1_정확성", "item2_핵심정보누락", "item3_지시형식준수",
                "item4_한국어표현", "item5_정보부족대응",
                "획득점", "만점", "환산100", "score_evidence", "answer"])
    for (m, q, run), v in sorted(S.items(), key=lambda kv: (kv[0][0], int(kv[0][1][1:]), kv[0][2])):
        i1, i2, i3, i4, i5, ev = v
        got = i1 + i2 + i3 + i4 + (i5 or 0)
        full = 10 if i5 is not None else 8
        r = idx[(m, q, run)]
        w.writerow([m, q, r["qtype"], run, i1, i2, i3, i4,
                    "" if i5 is None else i5,
                    got, full, round(got / full * 100, 1), ev,
                    " ".join((r["answer"] or "").split())])

# ── 집계 ──────────────────────────────────────────────────────────
import collections
agg = collections.defaultdict(list)
for (m, q, run), v in S.items():
    i1, i2, i3, i4, i5, _ = v
    got = i1 + i2 + i3 + i4 + (i5 or 0)
    full = 10 if i5 is not None else 8
    agg[m].append((q, got, full, [i1, i2, i3, i4, i5]))

with io.open("results/summary.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "n", "성공수", "전체시도수", "종합점수_100환산",
                "항목1_정확성", "항목2_핵심정보", "항목3_지시형식", "항목4_한국어",
                "항목5_정보부족(n=4)", "P4_Q9Q10_1점이상_횟수",
                "평균응답시간_s", "생성속도_tok_s", "VRAM_MiB"])
    for m in (Q, G):
        rs = agg[m]
        n = len(rs)
        pct = sum(g for _, g, _, _ in rs) / sum(fu for _, _, fu, _ in rs) * 100
        it = [[], [], [], [], []]
        for _, _, _, items in rs:
            for k in range(5):
                if items[k] is not None:
                    it[k].append(items[k])
        p4 = sum(1 for _, _, _, items in rs if items[4] is not None and items[4] >= 1)
        mr = [x for x in rows if x["model"] == m]
        w.writerow([m, n, n, n, round(pct, 1)]
                   + [round(sum(c) / len(c), 2) for c in it]
                   + [p4,
                      round(sum(x["elapsed_s"] for x in mr) / len(mr), 2),
                      round(sum(x["gen_tok_per_s"] for x in mr) / len(mr), 1),
                      mr[0]["vram_mib"]])

print("results/scores.csv, results/summary.csv 생성")
print()
for m in (Q, G):
    rs = agg[m]
    pct = sum(g for _, g, _, _ in rs) / sum(fu for _, _, fu, _ in rs) * 100
    kor = [items[3] for _, _, _, items in rs]
    p4 = sum(1 for _, _, _, items in rs if items[4] is not None and items[4] >= 1)
    print("%-14s 종합 %.1f점  한국어표현 %.2f/2  P4 %d/4" % (m, pct, sum(kor) / len(kor), p4))
