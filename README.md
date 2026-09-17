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
- **제약:** 수업용 Windows 노트북 1대, NVIDIA GeForce RTX 5060 Laptop GPU / VRAM 8,151 MiB
- **수행:** 단독 수행 (팀 분담 없음)

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
| Cloud | gpt-4o-mini | `gpt-4o-mini` | — | — | OpenAI API 이용약관 |

전체 비교표는 [docs/02_model_comparison.md](docs/02_model_comparison.md)에 있습니다.

### 2-1. 로컬 후보 모델 카드

| 실행 태그 | 원본 모델 | Model Size | Architecture | Context Length (카드 최대) | Language | GPU / VRAM (실측) | License |
|---|---|---|---|---|---|---|---|
| `qwen3.5:4b` | Qwen/Qwen3.5-4B | 4.7B | `qwen35` — Gated DeltaNet + sparse MoE 하이브리드, 32 layers, emb 2560 | 262,144 (YaRN 확장 시 ~1,010,000) | 201개 언어·방언 | 2,983 MiB / 100% GPU | Apache 2.0 |
| `qwen3.5:9b` | Qwen/Qwen3.5-9B | 9.7B | `qwen35` — Gated DeltaNet + sparse MoE 하이브리드, 32 layers, emb 4096 | 262,144 (YaRN 확장 시 ~1,010,000) | 201개 언어·방언 | 5,236 MiB / 100% GPU | Apache 2.0 |
| `gemma4:e2b` | google/gemma-4-E2B-it | 유효 2.3B / 전체 5.1B | `gemma4` — Dense + Per-Layer Embeddings, sliding window + global attention, 35 layers, emb 1536 | 131,072 | 기본 35개+ / 사전학습 140개+ | 1,629 MiB / 100% GPU | Apache 2.0 (+ Prohibited Use Policy) |
| `gemma4:e4b` | google/gemma-4-E4B-it | 유효 4.5B / 전체 8.0B | `gemma4` — Dense + Per-Layer Embeddings, sliding window + global attention, 42 layers, emb 2560 | 131,072 | 기본 35개+ / 사전학습 140개+ | 3,077 MiB / 100% GPU | Apache 2.0 (+ Prohibited Use Policy) |
| `exaone3.5:7.8b` | LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct | 7.8B | `exaone` — GQA (32 Q-head / 8 KV-head), 32 layers, vocab 102,400, emb 4096 | 32,768 | 한국어·영어 (2개국어) | 4,945 MiB / 100% GPU | **EXAONE AI Model License 1.1 - NC** |
| `hf.co/mradermacher/kanana-2-3b-instruct-GGUF:Q4_K_M` | kakaocorp/kanana-2-3b-instruct | 3.51B | `qwen3` (`Qwen3ForCausalLM`) — 32 layers, hidden 2560, GQA (32 Q-head / 8 KV-head), vocab 128,256 | 32,768 | 한국어·영어 | 2,462 MiB / 100% GPU | Kanana Open License Agreement |

**VRAM 측정 조건** — 각 모델을 1회 로드(`num_predict=1`)한 직후 Ollama `/api/ps`의 `size_vram` 값입니다.

- 측정 GPU: **NVIDIA GeForce RTX 5060 Laptop GPU, 8,151 MiB** / Ollama 0.34.0
- 측정 시 `context_length`는 Ollama 기본값 **4096**입니다. 실험에서 context를 키우면 KV 캐시만큼 증가하므로, 본 실험 값은 3절 설정으로 다시 측정해 기록합니다.
- `size_vram`은 **측정 시점의 값이며 최대 VRAM 사용량이 아닙니다.** 6개 모두 `size == size_vram`으로 100% GPU 적재되었습니다 (CPU 분할 없음).

**Gemma 4 E계열은 디스크 크기와 VRAM이 크게 다릅니다.** `gemma4:e2b`는 디스크 7.2 GB인데 VRAM은 1,629 MiB입니다. Per-Layer Embeddings 테이블이 GPU에 상주하지 않기 때문입니다. **디스크 크기 / 시스템 RAM / VRAM을 반드시 구분해 기록하십시오.**

Context Length는 모델 카드 최대값입니다. **실험에서 실제 사용한 `context_length`는 4절에 따로 기록합니다.**

### 2-2. 실행 식별값 (Ollama)

| 실행 태그 | digest | Quantization | 디스크 크기 | 모달리티 | Capabilities |
|---|---|---|---|---|---|
| `qwen3.5:4b` | `2a654d98e6fb` | Q4_K_M | 3.4 GB | Text / Image / Video | completion, vision, tools, thinking |
| `qwen3.5:9b` | `6488c96fa5fa` | Q4_K_M | 6.6 GB | Text / Image / Video | completion, vision, tools, thinking |
| `gemma4:e2b` | `7fbdbf8f5e45` | Q4_K_M | 7.2 GB | Text / Image / Audio | completion, vision, audio, tools, thinking |
| `gemma4:e4b` | `c6eb396dbd59` | Q4_K_M | 9.6 GB | Text / Image / Audio | completion, vision, audio, tools, thinking |
| `exaone3.5:7.8b` | `c7c4e3d1ca22` | Q4_K_M | 4.8 GB | Text | completion |
| `hf.co/mradermacher/kanana-2-3b-instruct-GGUF:Q4_K_M` | `d5a16a4a92bc` | `unknown` (태그명 기준 Q4_K_M) | 2.2 GB | Text | completion, tools, thinking |

digest·quantization·context·capabilities는 모두 로컬 `ollama list` / `ollama show` 출력에서 확인한 값입니다. Kanana 변환본만 GGUF 메타데이터에 quantization이 기록되지 않아 `unknown`으로 표시됩니다.

### 2-3. License

| 모델 | License | 상업적 사용 | 배포 시 의무 | P2 판정 |
|---|---|---|---|---|
| `qwen3.5:4b`, `qwen3.5:9b` | Apache License 2.0 | 가능 | 라이선스 사본 첨부, 변경사항 표시 | **통과** |
| `gemma4:e2b`, `gemma4:e4b` | Apache License 2.0 | 가능 | Apache 2.0 의무 + Google **Prohibited Use Policy** 준수 | **통과** |
| `hf.co/mradermacher/kanana-2-3b-instruct-GGUF:Q4_K_M` | Kanana Open License Agreement (Release Date 2025-07-17) | 조건부 가능 | **"Powered by Kanana" 표시** + Notice 파일 포함 | **통과** (아래 근거) |
| `exaone3.5:7.8b` | EXAONE AI Model License Agreement 1.1 - **NC** | **불가 (연구 목적 한정)** | 라이선스 사본 첨부, 파생 모델명에 "EXAONE" 접두 | **실패** |

**Kanana 2 3B — P2 통과 근거.** 4.1은 (i) API·클라우드 플랫폼 제공, (ii) SI·온프레미스 솔루션, (iii) 온디바이스 임베디드 적용에 대해 별도 상업 라이선스를 요구합니다. 본 Use Case(사내 상담 Assistant)는 이 세 가지에 해당하지 않는 **자체 서비스 개발·운영**이므로 4.2에 따라 별도 상업 라이선스가 필요하지 않습니다. 단 배포 시 3.1(v)의 "Powered by Kanana" 표시와 Notice 파일 포함 의무가 발생합니다. 또한 2.4~2.5에 따라 **버전별 라이선스가 독립**이므로, Kanana 1.5의 Apache 2.0 조건을 Kanana 2에 적용할 수 없습니다.

**EXAONE 3.5 — P2 실패 근거.** 라이선스 3.1은 *"any commercial purposes, including but not limited to, developing or deploying products, services, or applications that generate revenue"* 를 금지하고, 2.1(a)는 사용 범위를 *"solely for research purposes... evaluation, testing, academic research, experimentation"* 으로 한정합니다. 본 Use Case는 사내 상담 서비스 운영이므로 **필수 통과 조건 P2를 충족하지 못합니다.** 따라서 **최종 선정 후보에서 제외**하고, 라이선스 사유로 탈락한 사례로만 기록합니다. (2.1(a)가 평가·테스트 목적의 실행 자체는 허용하므로 참고 측정값은 남길 수 있습니다.)

### 2-4. Kanana 2 3B 사용 시 주의

**공식 GGUF가 없습니다.** 카카오는 Kanana 2 3B의 GGUF 배포본을 제공하지 않아, 커뮤니티 변환본 [`mradermacher/kanana-2-3b-instruct-GGUF`](https://huggingface.co/mradermacher/kanana-2-3b-instruct-GGUF)를 사용했습니다. 공식 배포본과 양자화·템플릿 구성이 다를 수 있으므로, **본 실험 결과는 해당 변환본 기준이며 원본 모델의 성능과 동일하다고 단정하지 않습니다.** 실제로 `ollama show --license` 출력이 비어 있어(변환본에 LICENSE 미포함) 라이선스는 원본 저장소 기준으로 확인했습니다. **또한 이 변환본에는 채팅 템플릿도 포함되어 있지 않아**(`TEMPLATE {{ .Prompt }}`) 첫 실행이 비정상 동작했습니다. 템플릿 복원 과정과 복원 전후 비교는 2-8을 참조하십시오.

**베이스 모델 확인 결과 — Qwen3에서 이어서 학습한 모델이 아닙니다.** `config.json`의 `model_type`은 `"qwen3"`, `architectures`는 `["Qwen3ForCausalLM"]`로 **Qwen3 아키텍처 클래스를 그대로 사용**합니다. 그러나 모델 카드는 *"Kanana-2-3B was pretrained from scratch on TPU clusters"* 라고 명시하고, `vocab_size`도 128,256으로 Qwen3와 다른 자체 토크나이저를 씁니다. 즉 **가중치를 물려받은 파생 모델이 아니라, 같은 아키텍처 정의로 처음부터 사전학습한 모델**입니다.

**따라서 비교 해석은 이렇게 기술합니다.** `qwen3.5:4b`(디스크 3.4 GB / VRAM 2,983 MiB)와 Kanana 2 3B(디스크 2.2 GB / VRAM 2,462 MiB)는 크기가 가까워, **크기 변수가 어느 정도 통제된 상태에서 "한국어 중심 학습 데이터가 실제로 유리한가"를 물을 수 있습니다.** 다만 Qwen3.5는 Gated DeltaNet + sparse MoE(`qwen35`)로 Kanana 2의 `qwen3` 구조와 다릅니다. **아키텍처가 통제된 비교는 아니며**, 이를 한계로 명시합니다.

### 2-5. Cloud 비교 모델 — `gpt-4o-mini`

| 항목 | 값 |
|---|---|
| 모델 | `gpt-4o-mini` (OpenAI API) |
| Context Length | 128,000 |
| 최대 출력 | 16,384 |
| 모달리티 | Text + Image 입력 / Text 출력 |
| 지식 컷오프 | 2023-10 |
| 단가 | Input $0.15 / 1M · Cached input $0.075 / 1M · Output $0.60 / 1M |
| 추론(thinking) 토큰 | 없음 |
| License | 오픈 웨이트 아님 — OpenAI API 이용약관 적용 |

**선정 이유**

- **비용이 현실적입니다.** 질문 5개 × 1회면 몇 센트 수준이고, 테스트하다 호출이 늘어도 부담이 없습니다. "실제로 도입한다면 이 정도 단가를 쓰겠다"는 계산이 성립합니다. 플래그십은 상담 봇에 붙이기엔 단가가 맞지 않아 비용 비교 자체가 비현실적이 됩니다.
- **격차가 적당합니다.** 로컬 3~9B와 붙였을 때 품질 차이는 분명히 나되, "그래도 로컬을 쓸 이유가 있나"를 따질 만한 수준입니다. 초소형이면 Cloud를 쓸 이유가 없어지고, 최상위면 결론이 뻔해집니다.
- **변수가 없습니다.** 안정적이고 응답이 빠르며, 추론 토큰이 끼어들지 않아 출력 길이와 응답 시간 측정이 깨끗합니다. thinking 모델을 Cloud 쪽에 쓰면 로컬과 비교 기준이 또 달라집니다.
- **자료가 많아 토큰 단가를 찾기 쉽고**, 그 단가가 비용 계산 근거로 바로 들어갑니다.

### 2-6. 검토 후 제외한 모델

| 모델 계열 | 제외 사유 |
|---|---|
| DeepSeek | **용도가 다릅니다.** 수학·코딩 추론 벤치마크를 겨냥한 reasoning 특화 계열입니다. 본 Use Case는 문서 근거에 기반한 짧은 한국어 상담 답변이라 목적이 맞지 않고, 긴 사고 토큰이 응답 시간·출력 토큰 측정을 왜곡합니다. |
| Kimi | **용도가 다릅니다.** 초장문 컨텍스트와 에이전트 작업을 겨냥한 대형 계열로, Use Case와 수업용 노트북(VRAM 8,151 MiB) 실행 범위 양쪽에 맞지 않습니다. |

### 2-7. Model Card / License 원문 링크

| 대상 | 링크 |
|---|---|
| Qwen3.5-4B Model Card | https://huggingface.co/Qwen/Qwen3.5-4B |
| Qwen3.5-9B Model Card | https://huggingface.co/Qwen/Qwen3.5-9B |
| Qwen3.5 License 원문 (Apache 2.0) | https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE |
| Qwen3.5 GitHub | https://github.com/QwenLM/Qwen3.5 |
| Qwen3.5 Ollama 페이지 | https://ollama.com/library/qwen3.5 |
| Gemma 4 E2B Model Card | https://huggingface.co/google/gemma-4-E2B-it |
| Gemma 4 E4B Model Card | https://huggingface.co/google/gemma-4-E4B-it |
| Gemma 4 License 원문 | https://ai.google.dev/gemma/docs/gemma_4_license |
| Gemma 4 Prohibited Use Policy | https://ai.google.dev/gemma/prohibited_use_policy |
| Gemma 4 공식 문서 | https://ai.google.dev/gemma/docs/core |
| Gemma 4 Technical Report | https://arxiv.org/abs/2607.02770 |
| Gemma 4 Ollama 페이지 | https://ollama.com/library/gemma4 |
| EXAONE 3.5 7.8B Model Card | https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct |
| EXAONE 3.5 License 원문 (1.1 - NC) | https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct/blob/main/LICENSE |
| EXAONE 3.5 Ollama 페이지 | https://ollama.com/library/exaone3.5 |
| Kanana 2 3B Instruct Model Card | https://huggingface.co/kakaocorp/kanana-2-3b-instruct |
| Kanana Open License 원문 | https://huggingface.co/kakaocorp/kanana-2-3b-instruct/blob/main/LICENSE |
| Kanana 2 3B GGUF 변환본 (실제 사용) | https://huggingface.co/mradermacher/kanana-2-3b-instruct-GGUF |
| gpt-4o-mini 모델 문서 | https://developers.openai.com/api/docs/models/gpt-4o-mini |
| OpenAI API 가격 | https://developers.openai.com/api/docs/pricing |

조사·측정 기준일: 2026-09-16. VRAM은 RTX 5060 Laptop GPU(8,151 MiB), Ollama 0.34.0, `context_length=4096` 기준 실측값입니다.

### 2-8. 후보 선별 예비 실행 (본 실험 아님)

> **이 표는 필수 본 실험이 아닙니다.** 로컬 A·B를 고르기 위한 STEP 3 후보 선별용 예비 실행이며, **모델 5개 × 질문 10개 × 1회 = 50회**입니다.
> 필수 본 실험은 **최종 후보 2개 × 10개 질문 × 2회 = 40회**로 별도 수행하며, 아래 결과는 **본 실험 집계에 포함하지 않습니다.**

**실행 조건** (모든 모델 동일) — `num_ctx=4096`, `temperature=0.2`, `think=False`, 대화 이력 없음(단발 질의), 모델당 워밍업 1회 선행 후 분리.

**판정 기준** — 아래 ✅/◐/❌는 후보 선별을 위한 **간이 판정**이며, [data/rubric.md](data/rubric.md)의 5항목 × 0~2점 정식 채점이 아닙니다. 정식 채점은 본 실험 결과에만 적용합니다.

| 표기 | 의미 |
|---|---|
| ✅ | 기대 결과와 일치 |
| ◐ | 방향은 맞으나 금액 미계산·조건 일부 누락 |
| ❌ | 오답, 문서에 있는 내용인데 회피, 또는 문서에 없는 내용 생성 |

#### 문항별 판정

| 모델 | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 | **Q9** | **Q10** | ✅ | ◐ | ❌ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `gemma4:e4b` | ◐ | ❌ | ◐ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **7** | 2 | 1 |
| `qwen3.5:9b` | ◐ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ◐ | ✅ | ✅ | **6** | 2 | 2 |
| `qwen3.5:4b` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ◐ | ✅ | 6 | 1 | 3 |
| `gemma4:e2b` | ◐ | ◐ | ❌ | ◐ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | 5 | 3 | 2 |
| Kanana 2 3B (템플릿 복구) | ❌ | ❌ | ❌ | ◐ | ✅ | ❌ | ✅ | ◐ | ❌ | ❌ | 2 | 2 | **6** |

**Q9·Q10은 P4 판정 문항입니다.** Kanana만 두 문항 모두 문서에 없는 내용을 생성했습니다.

#### 요약

| 모델 | 응답 반환 | 평균 응답 시간 | 평균 출력 | VRAM | License | Q9·Q10 | 선별 결과 |
|---|---|---|---|---|---|---|---|
| `gemma4:e4b` | 10/10 | 0.89s | 66 tok | 3,077 MiB | Apache 2.0 | 2/2 ✅ | **로컬 후보** |
| `qwen3.5:9b` | 10/10 | 1.65s | 71 tok | 5,236 MiB | Apache 2.0 | 2/2 ✅ | **로컬 후보** |
| `gemma4:e2b` | 10/10 | 0.55s | 70 tok | 1,629 MiB | Apache 2.0 | 2/2 ✅ | 예비 |
| `qwen3.5:4b` | 10/10 | 1.03s | 70 tok | 2,983 MiB | Apache 2.0 | 2/2 ✅ | 제외 — 오답 3건, 중국어 토큰 혼입(`出发`) |
| Kanana 2 3B | 10/10 | 0.19s | 20 tok | 2,462 MiB | Kanana Open License | **0/2 ❌** | **제외 — P4 근거 미달** |
| `exaone3.5:7.8b` | — | — | — | 4,945 MiB | **1.1 - NC** | — | **제외 — P2 실패 (미실행)** |

P4 정식 판정은 본 실험 Q9·Q10 **4회** 기준입니다. 위는 1회 실행 결과이므로 정식 판정이 아니라 **후보 선별 근거**입니다.

#### 대표 실패 사례 — 실패 원인이 모델이 아니라 환경이었던 경우

Kanana 2 3B 커뮤니티 변환본의 첫 실행 결과가 비정상적으로 짧았습니다(평균 19토큰). 원인을 추적한 결과 **GGUF에 채팅 템플릿이 포함되어 있지 않았습니다.**

```
$ ollama show hf.co/mradermacher/kanana-2-3b-instruct-GGUF:Q4_K_M --modelfile
TEMPLATE {{ .Prompt }}
```

역할 마커(`<|im_start|>` / `<|im_end|>`) 없이 원문이 그대로 입력되고 있었습니다. 원본 저장소의 `chat_template.jinja`에서 ChatML 구조와 기본 `no_think` 모드를 확인해 템플릿을 복원하고 `kanana2-3b-chatml:q4km`으로 재빌드한 뒤 재측정했습니다.

| 문항 | 템플릿 누락 상태 | 템플릿 복구 후 |
|---|---|---|
| Q5 (1인 객실 추가 요금) | ❌ "추가 요금이 붙습니다" (금액 없음) | ✅ "220,000원" |
| Q7 (천재지변 면제) | ❌ "확인이 어렵습니다" | ✅ 면제 + 증빙 + 예외까지 |
| **Q9 (보험 한도)** | ❌ "보험 약관에 따라 다릅니다" | ❌ **"3천만원입니다"** |
| **Q10 (베트남 다낭)** | ❌ "90일 이내 비자 면제" | ❌ **"90일 이내 비자 면제"** |

**실패 원인 구분:** Q5·Q7은 **환경(변환본 템플릿 누락)** 원인이었고 복구 후 해소되었습니다. Q9·Q10은 템플릿 복구 후에도 남았으므로 **모델** 원인입니다. 따라서 이 변환본의 P4 미달은 변환 결함으로 설명되지 않습니다.

이 결과는 **해당 커뮤니티 변환본 기준**이며 원본 `kakaocorp/kanana-2-3b-instruct`의 성능과 동일하다고 단정하지 않습니다 (2-4 참조).

#### 선별 결론

- **한국어 특화 소형 모델 축은 이 프로젝트에서 성립하지 않습니다.** 후보 2개 중 EXAONE 3.5는 P2(License), Kanana 2 3B는 P4(근거 없는 생성)에서 각각 필수 조건을 통과하지 못했습니다.
- 로컬 A·B는 필수 조건을 통과한 후보 중 **실패 방식이 서로 다른** 두 모델로 선정합니다.
  - `qwen3.5:9b` — 금액을 실제로 계산하는 유일한 후보 (Q2 534,000원, Q3 712,000원). 대신 문서에 명시된 내용을 **과잉 회피**하는 경향 (Q4·Q6).
  - `gemma4:e4b` — 문서 이해와 지시 준수가 안정적이고 경계 사례(Q8) 처리가 가장 정확. 대신 **금액 계산을 회피** (Q2·Q3).

#### 필수 실행 설정

qwen3.5 계열은 **thinking이 기본 활성**입니다. 기본값으로 `num_ctx=4096`에서 실행하면 thinking이 컨텍스트를 모두 소진해 **빈 응답을 반환합니다** (실측: 62.2초, prompt 979 + output 3,119 = 4,096 소진, 본문 없음). 따라서 본 실험은 **모든 모델에 `think=False`를 동일 적용**합니다.

| 설정 | qwen3.5:9b 실측 |
|---|---|
| `num_ctx=4096`, think **on** | 62.2s, **빈 응답** (P1 실패) |
| `num_ctx=8192`, think on | 77.2s, 응답 정상, **88% GPU / 12% CPU** |
| `num_ctx=4096`, think **off** | **7.5s, 응답 정상, 100% GPU** |

---

## 3. 실험 설계

| 항목 | 값 |
|---|---|
| 고정 질문 | 10개 (정상 6 / 경계 2 / 정보 부족·범위 밖 2) |
| 로컬 본 실험 | 2개 모델 × 10개 질문 × 2회 = **40회** |
| 로컬 워밍업 | 모델당 1회 (본 실험 집계에서 분리) |
| Cloud 본 실험 | 1개 모델 × 5개 질문 × 1회 = **5회** |
| 채점 | 5항목 × 0~2점, 100점 환산 집계 |

질문 세트와 채점 기준은 **첫 호출 이전에 확정·커밋**했습니다. (커밋 [`7800589`](https://github.com/jeongrolee-tech/local-llm-travel-assistant/commit/7800589))

- 정책 문서: [data/policy.md](data/policy.md)
- 질문 10개: [data/questions.json](data/questions.json)
- 채점 루브릭: [data/rubric.md](data/rubric.md)
- 고정 시스템 지시문: [data/system_prompt.txt](data/system_prompt.txt)

---

## 4. 실행 환경

| 항목 | 값 |
|---|---|
| OS | Windows 11 Pro 10.0.26200 |
| Python | 3.12 |
| 패키지 관리 | uv |
| Ollama | 0.34.0 |
| GPU / VRAM | NVIDIA GeForce RTX 5060 Laptop GPU / 8,151 MiB |
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

## 10. 저장소에 포함하지 않는 것

- 모델 가중치 파일
- API 키 및 `.env`
- 가상환경 디렉터리 전체
- 실제 고객 데이터 (본 실험은 가상 문서만 사용)
