# local-llm-travel-assistant

한국어 여행 상담 Assistant에 사용할 로컬 LLM을 선정하기 위한 비교 실험 저장소입니다.

> **결론:** `{모델명}` 을 선정했습니다. `{핵심 근거 한 줄 — 예: 필수 조건 4개를 모두 통과한 유일한 후보이며, 문서에 없는 규정을 지어내지 않았습니다.}`
>
> 상세 근거는 [최종 선정 보고서](docs/05_selection_report.md)를 참고하십시오.

---

## 1. 문제 정의

가상 여행사 ○○투어의 패키지 상품 안내서와 취소·환불 규정 문서를 근거로, 고객의 한국어 문의에 답변하는 상담 Assistant에 사용할 로컬 모델 1개를 선정합니다.

- **사용자:** 상담 직원 (Assistant가 초안을 만들고 직원이 확인 후 발송)
- **중요 가치:** 한국어 표현 자연스러움 > 데이터 보안 > 응답 속도
- **제약:** 수업용 Windows 노트북 1대, `{GPU 모델명 / VRAM}`

### 필수 통과 조건

| 코드 | 조건 |
|---|---|
| P1 | 수업용 노트북에서 실행되고 질문 10개 전체에 응답 반환 |
| P2 | 상업적 사용이 가능한 License |
| P3 | 문서 + 질문 + 답변이 실험 Context Length 내 수용 |
| P4 | 문서에 없는 규정을 지어내지 않음 (Q9·Q10 4회 중 3회 이상 통과) |

### 선호 우선순위

종합 품질 점수 → 한국어 표현 점수 → 전체 응답 시간 → VRAM 사용량

---

## 2. 비교 대상

| 구분 | 모델 | 실행 태그 | digest | Quantization | License |
|---|---|---|---|---|---|
| 로컬 A | `{}` | `{}` | `{}` | `{}` | `{}` |
| 로컬 B | `{}` | `{}` | `{}` | `{}` | `{}` |
| Cloud | `{}` | — | — | — | `{}` |

전체 비교표는 [docs/02_model_comparison.md](docs/02_model_comparison.md)에 있습니다.

---

## 3. 실험 설계

| 항목 | 값 |
|---|---|
| 고정 질문 | 10개 (정상 6 / 경계 2 / 정보 부족·범위 밖 2) |
| 로컬 본 실험 | 2개 모델 × 10개 질문 × 2회 = **40회** |
| 로컬 워밍업 | 모델당 1회 (본 실험 집계에서 분리) |
| Cloud 본 실험 | 1개 모델 × 5개 질문 × 1회 = **5회** |
| 채점 | 5항목 × 0~2점, 100점 환산 집계 |

질문 세트와 채점 기준은 **첫 호출 이전에 확정·커밋**했습니다. (커밋 `{해시}`)

- 정책 문서: [data/policy.md](data/policy.md)
- 질문 10개: [data/questions.json](data/questions.json)
- 채점 루브릭: [data/rubric.md](data/rubric.md)
- 고정 시스템 지시문: [data/system_prompt.txt](data/system_prompt.txt)

---

## 4. 실행 환경

| 항목 | 값 |
|---|---|
| OS | Windows `{버전}` |
| Python | 3.12 |
| 패키지 관리 | uv |
| Ollama | `{버전}` |
| GPU / VRAM | `{}` |
| 시스템 RAM | `{}` |

생성 설정은 모든 모델·모든 회차에 동일하게 적용했습니다.

| 설정 | 값 |
|---|---|
| temperature | `{}` |
| 출력 한도 | `{}` |
| context_length | `{}` |
| 대화 이력 | 매 호출 초기화 (단발 질의) |

---

## 5. 실행 방법

```bash
# 1. 의존성 설치
uv sync

# 2. Ollama 모델 준비
ollama pull {모델A 태그}
ollama pull {모델B 태그}

# 3. 로컬 실험 (워밍업 1회 후 본 실험 20회)
uv run python src/run_local.py --model {모델A 태그}
uv run python src/run_local.py --model {모델B 태그}

# 4. Cloud 실험
uv run python src/run_cloud.py

# 5. 집계
uv run python src/aggregate.py
```

API 키는 `.env`로 관리하며 저장소에 포함하지 않습니다. `.env.example`을 복사해 사용하십시오.

---

## 6. 결과 파일 위치

| 파일 | 내용 |
|---|---|
| `results/local_raw.jsonl` | 로컬 본 실험 40회 원본 응답과 측정값 |
| `results/local_warmup.jsonl` | 워밍업 기록 (집계 제외) |
| `results/cloud_raw.jsonl` | Cloud 5회 원본 응답, 토큰, 비용 |
| `results/scores.csv` | 회차별 채점 결과와 점수 근거 |
| `results/summary.csv` | 모델별 집계 (평균, n, 성공 수/전체 시도 수) |
| `results/errors.jsonl` | 호출 실패·측정 불가 기록과 사유 |

각 회차 레코드에는 질문 ID, 반복 회차, 모델 태그, digest, quantization_level, context_length, 응답 시간, 로딩 시간, 출력 토큰 수, 생성 속도, VRAM, 성공·오류 상태를 기록합니다.

---

## 7. 결과 요약

| 모델 | 종합 점수 (100점 환산) | n | 성공/시도 | 평균 응답 시간 | 생성 속도 | VRAM | 필수 조건 |
|---|---|---|---|---|---|---|---|
| 로컬 A | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{P1~P4 판정}` |
| 로컬 B | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |
| Cloud | `{}` | `{}` | `{}` | `{}` | — | — | 비교 기준 |

Cloud는 질문 5개 × 1회, 로컬은 10개 × 2회로 **반복 수가 다릅니다.** 직접 순위 비교의 근거로 사용하지 않았습니다.

---

## 8. 한계

- 질문 10개, 반복 2회의 소규모 평가셋입니다.
- 단일 노트북 1대에서 측정한 값이므로 다른 장비의 절대 속도와 합산하지 않았습니다.
- `size_vram`은 측정 시점의 값이며 최대 VRAM 사용량이 아닙니다.
- `ollama ps`의 PROCESSOR는 CPU/GPU 적재 상태이며 GPU 이용률이 아닙니다.

---

## 9. 문서

| 문서 | 내용 |
|---|---|
| [docs/02_model_comparison.md](docs/02_model_comparison.md) | Model Comparison Table |
| [docs/03_benchmark.md](docs/03_benchmark.md) | 실험 결과와 품질 평가 |
| [docs/04_local_vs_cloud.md](docs/04_local_vs_cloud.md) | Local LLM vs Cloud API 비교 |
| [docs/05_selection_report.md](docs/05_selection_report.md) | 최종 선정 보고서 |

---

## 10. 참여

| 이름 | 역할 | 담당 |
|---|---|---|
| `{}` | 프로젝트 리드·요구사항 | Use Case 정의, 필수 조건·우선순위 확정, 최종 선정 |
| `{}` | 모델 리서치·환경 | 후보 조사, Model Card·License 확인, 실행 환경 구성 |
| `{}` | 실험·성능 측정 | 로컬 40회·Cloud 5회 실행, 측정값 수집 |
| `{}` | 품질 평가·문서화 | 채점, 실패 사례 분석, 저장소 정리 |

> 개인 수행인 경우 이 표를 삭제하고 단독 수행임을 명시하십시오.

---

## 11. 저장소에 포함하지 않는 것

- 모델 가중치 파일
- API 키 및 `.env`
- 가상환경 디렉터리 전체
- 실제 고객 데이터 (본 실험은 가상 문서만 사용)
