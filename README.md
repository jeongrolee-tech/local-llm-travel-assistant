# local-llm-travel-assistant

한국어 여행 상담 Assistant에 사용할 로컬 LLM을 선정하기 위한 비교 실험 저장소입니다.

> **결론:** **`gemma4:e4b`** 를 선정했습니다. 두 후보 모두 필수 조건 4개를 통과했고 종합 점수 차이가 5점 이내라 동점으로 보았으며, 2순위인 **한국어 표현 점수에서 앞섰습니다**(1.95 vs 1.65). 응답 시간과 VRAM에서도 유리합니다.
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
| 로컬 A | Qwen3.5 9B | `qwen3.5:9b` | `6488c96fa5fa` | Q4_K_M | Apache 2.0 |
| 로컬 B | Gemma 4 E4B | `gemma4:e4b` | `c6eb396dbd59` | Q4_K_M | Apache 2.0 (+ Prohibited Use Policy) |
| Cloud | gpt-4o-mini | `gpt-4o-mini` | — | — | OpenAI API 이용약관 |

전체 비교표는 [docs/02_model_comparison.md](docs/02_model_comparison.md)에 있습니다. 두 후보는 **2-8의 후보 선별 예비 실행**을 근거로 선정했습니다. 선정 축은 *실패 방식이 서로 다른 두 모델* 입니다 — 로컬 A는 금액을 계산하지만 문서에 있는 내용을 과잉 회피하고, 로컬 B는 문서 이해·지시 준수가 안정적이지만 금액 계산을 회피합니다.

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

#### Tokenizer — 한국어 입력 토큰 수 실측

| 모델 | Vocab Size | 동일 입력 프롬프트 토큰 | Kanana 대비 |
|---|---|---|---|
| Kanana 2 3B | 128,256 | **811.2** | 기준 |
| `qwen3.5:9b` / `qwen3.5:4b` | 248,320 | 979.6 | **+20.8%** |
| `gemma4:e4b` / `gemma4:e2b` | 262,144 | 1,071.6 | **+32.1%** |
| `exaone3.5:7.8b` | 102,400 | 미측정 (P2 실패로 미실행) | — |

입력은 전 모델 동일합니다 — `data/system_prompt.txt` + `data/policy.md` + 질문 10개. 예비 실행 10회 평균이며 회차별 원본은 [docs/pilot/](docs/pilot/)의 `.jsonl`에 있습니다.

같은 한국어 문서를 Kanana는 811 토큰, Gemma 4는 1,072 토큰으로 표현합니다. **한국어 특화 토크나이저의 효율은 실측으로 확인되지만, 본 실험에서 그 이점이 품질로 이어지지는 않았습니다** — Kanana는 P4에서 탈락했습니다(2-8).

#### Chat Template

| 모델 | 적용 방식 | `ollama show --modelfile` |
|---|---|---|
| `qwen3.5:9b` / `qwen3.5:4b` | 내장 렌더러 | `TEMPLATE {{ .Prompt }}` + `RENDERER qwen3.5` + `PARSER qwen3.5` |
| `gemma4:e4b` / `gemma4:e2b` | 내장 렌더러 | `TEMPLATE {{ .Prompt }}` + `RENDERER gemma4` + `PARSER gemma4` |
| `exaone3.5:7.8b` | Go 템플릿 | `.Messages` 순회 + `PARAMETER stop [\|endofturn\|]` + `SYSTEM` |
| Kanana 2 3B (커뮤니티 변환본) | **GGUF 내장 Jinja 템플릿** | `TEMPLATE {{ .Prompt }}` — 그러나 `--template`은 공식 Jinja 10,725 bytes를 출력 |

**`ollama show --modelfile`의 `TEMPLATE` 줄로는 템플릿 유무를 판단할 수 없습니다.** 네 모델 모두 `TEMPLATE {{ .Prompt }}`를 출력하지만, 실제 대화 형식은 Qwen3.5·Gemma 4의 경우 `RENDERER`/`PARSER`가, Kanana 변환본의 경우 **GGUF에 내장된 공식 Jinja 템플릿**이 처리합니다. 확인하려면 `ollama show --template`을 써야 합니다. 이 점을 처음에 잘못 판단했고 재검증으로 바로잡은 과정은 2-8에 기록했습니다.

#### 공개 Benchmark

모델 카드에 게시된 점수이며 **직접 측정한 값이 아닙니다.**

| 벤치마크 | `qwen3.5:9b` | `gemma4:e4b` | `qwen3.5:4b` | `gemma4:e2b` | `exaone3.5:7.8b` | Kanana 2 3B |
|---|---|---|---|---|---|---|
| MMLU-Pro | **82.5** | 69.4 | 79.1 | 60.0 | 46.24 | — (MMLU-CoT 61.09) |
| GPQA Diamond | 81.7 | 58.6 | 76.2 | 43.4 | — | — |
| MMMLU (다국어) | 81.2 | 76.6 | 76.1 | 67.4 | — | — |
| IFEval | 91.5 | — | 89.8 | — | 78.9 | 80.96 |
| **KMMLU (CoT)** | 미공개 | 미공개 | 미공개 | 미공개 | 미공개 | **43.32** |
| **KoMT-Bench** | 미공개 | 미공개 | 미공개 | 미공개 | **7.96** | 6.92 |
| **HAE-RAE Bench** | 미공개 | 미공개 | 미공개 | 미공개 | 미공개 | 43.75 |

**이 표로 순위를 매기지 않습니다.** 모델마다 공개 항목과 평가 프로토콜이 다르고, Qwen3.5 점수는 thinking 모드 기준일 가능성이 높은데 본 실험은 `think=False`입니다(4절).

**결정적으로, 로컬 A·B로 선정한 Qwen3.5와 Gemma 4는 한국어 단독 벤치마크를 공개하지 않았습니다.** 한국어 지표를 공개한 것은 EXAONE 3.5(P2 실패)와 Kanana 2(P4 실패)뿐입니다. 공개 벤치마크만으로는 이 Use Case의 적합성을 판단할 수 없으며, 고정 질문 10개로 자체 측정하는 근거가 여기에 있습니다.

전체 비교표는 [docs/02_model_comparison.md](docs/02_model_comparison.md)에 있습니다.

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

**공식 GGUF가 없습니다.** 카카오는 Kanana 2 3B의 GGUF 배포본을 제공하지 않아, 커뮤니티 변환본 [`mradermacher/kanana-2-3b-instruct-GGUF`](https://huggingface.co/mradermacher/kanana-2-3b-instruct-GGUF)를 사용했습니다. 공식 배포본과 양자화·템플릿 구성이 다를 수 있으므로, **본 실험 결과는 해당 변환본 기준이며 원본 모델의 성능과 동일하다고 단정하지 않습니다.** 실제로 `ollama show --license` 출력이 비어 있어(변환본에 LICENSE 미포함) 라이선스는 원본 저장소 기준으로 확인했습니다. 반면 **채팅 템플릿은 정상적으로 포함되어 있습니다** — `ollama show --template`이 원본 저장소와 같은 공식 Jinja 템플릿(10,725 bytes)을 출력합니다. 이를 누락으로 오판했다가 재검증한 과정은 2-8에 있습니다.

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

#### 대표 실패 사례 — 원인을 환경으로 오판했다가 재검증으로 바로잡은 경우

Kanana 2 3B 커뮤니티 변환본의 첫 실행 결과가 비정상적으로 짧았습니다(평균 19토큰, 대부분 오답).

**1차 가설 — 채팅 템플릿 누락 (틀렸습니다).** `ollama show --modelfile` 출력이 `TEMPLATE {{ .Prompt }}` 한 줄뿐이라 대화 형식이 적용되지 않는다고 판단했습니다. 원본 저장소의 `chat_template.jinja`를 참고해 ChatML Go 템플릿을 만들고 `kanana2-3b-chatml:q4km`으로 재빌드해 재측정했더니 Q5·Q7이 개선된 것처럼 보였고, 이를 **환경 원인**으로 기록했습니다.

**재검증 — 가설이 기각되었습니다.** 세 가지가 어긋났습니다.

| 검증 | 결과 | 의미 |
|---|---|---|
| 재빌드 전후 프롬프트 토큰 비교 | 10문항 **전부 diff = 0** (811, 818, 809, …) | 모델에 들어간 입력이 바뀌지 않았다 |
| `temperature=0`, `seed=42`로 `"안녕"` 1개 메시지 | 양쪽 모두 `prompt_eval_count=14`, 답변 문자열 동일 | 재빌드가 아무것도 바꾸지 않았다 |
| `ollama show --template` | **공식 Jinja 템플릿 10,725 bytes 출력** (원본 저장소 파일과 동일 크기) | 템플릿은 처음부터 GGUF에 내장되어 정상 적용되고 있었다 |

`--modelfile`의 `TEMPLATE {{ .Prompt }}`는 Go 템플릿 필드 표시일 뿐이고, 실제 형식은 GGUF 내장 Jinja가 처리하고 있었습니다. **재빌드는 무효과였고, Q5·Q7의 차이는 `temperature=0.2` 단일 실행의 샘플링 변동이었습니다.**

`temperature=0`, `seed=42`로 고정해 원본 변환본을 다시 돌려 확인했습니다.

| 문항 | 1차 실행 (temp 0.2) | 2차 실행 (temp 0.2) | 재검증 (temp 0) | 판정 |
|---|---|---|---|---|
| Q5 (1인 객실 추가 요금) | "추가 요금이 붙습니다" | "220,000원입니다" | **"추가 요금이 붙습니다"** | 샘플링 변동 |
| Q7 (천재지변 면제) | "확인이 어렵습니다" | 면제 + 증빙 + 예외 | **면제 + 증빙 + 예외** | 샘플링 변동 |
| **Q9 (보험 한도)** | 문서 밖 생성 | "3천만원입니다" | **문서 밖 생성** | **모델 원인 — 일관됨** |
| **Q10 (베트남 다낭)** | "90일 비자 면제" | "90일 비자 면제" | **"90일 비자 면제"** | **모델 원인 — 일관됨** |

**실패 원인 구분 — 정정된 결론.** Q5·Q7은 환경이 아니라 **측정**의 문제였습니다(1회 실행의 샘플링 변동을 설정 변경 효과로 오독). Q9·Q10은 `temperature=0`에서도 동일하게 재현되므로 **모델** 원인입니다. **Kanana 2 3B의 P4 미달은 변환본 결함이 아니라 모델 자체의 결과이며, 탈락 판정은 유지됩니다.**

**여기서 얻은 교훈 두 가지를 실험 규칙으로 반영합니다.**

1. `ollama show --modelfile`의 `TEMPLATE` 줄로 템플릿 유무를 판단하지 않습니다. `--template`으로 확인합니다.
2. `temperature > 0`인 **1회 실행의 응답 차이를 설정 변경의 효과로 해석하지 않습니다.** 본 실험이 질문당 2회를 요구하는 이유가 여기에 있습니다.

재검증 원본은 [docs/pilot/04_kanana-recheck-temp0.txt](docs/pilot/04_kanana-recheck-temp0.txt)에 있습니다. 이 결과는 **해당 커뮤니티 변환본 기준**이며 원본 `kakaocorp/kanana-2-3b-instruct`의 성능과 동일하다고 단정하지 않습니다 (2-4 참조).

#### 모델 크기와 품질의 관계

| 모델 | 파라미터 | ✅ | ❌ | P4 (Q9·Q10) |
|---|---|---|---|---|
| `qwen3.5:9b` | 9.7B | 6 | 2 | 2/2 ✅ |
| **`gemma4:e4b`** | **유효 4.5B** | **7** | **1** | 2/2 ✅ |
| `qwen3.5:4b` | 4.7B | 6 | 3 | 2/2 ✅ |
| `gemma4:e2b` | 유효 2.3B | 5 | 2 | 2/2 ✅ |
| Kanana 2 3B | 3.51B | 2 | 6 | **0/2 ❌** |

**크기 순서와 성적 순서가 일치하지 않습니다.** 로컬 B로 선정한 `gemma4:e4b`는 유효 4.5B로 `qwen3.5:9b`(9.7B)보다 작지만 ✅가 더 많았습니다. 유효 2.3B~9.7B 구간에서 ✅는 5~7로 크게 벌어지지 않습니다.

**환각 방지(P4)도 크기로 설명되지 않습니다.** 4B급을 포함해 4개 모두 통과했고, 문서에 없는 내용을 생성한 것은 중간 크기인 Kanana 2 3B(3.51B) 하나뿐입니다.

**크기가 작아질 때 실제로 나타난 것은 환각 증가가 아니라 표현·형식 불안정입니다.**

| 모델 | 문항 | 관찰 | 루브릭 영향 |
|---|---|---|---|
| `qwen3.5:4b` | Q1 | "출발 25일 전 취소 시에는 **出发** 30일 전까지와 동일하게…" — 중국어 토큰 혼입 | 항목 4(한국어 표현) **0점 기준** — "외국어 혼입" |
| `gemma4:e2b` | Q3, Q7 | 정답을 말한 뒤 "제공된 자료로는 확인이 어렵습니다"를 덧붙여 자기모순 | 항목 3(지시·형식 준수) 불안정 |

본 프로젝트의 중요 가치 1순위가 **한국어 표현 자연스러움**이고 선호 우선순위 2순위도 한국어 표현 점수입니다. 소형화의 대가가 정확히 이 지점에서 나타납니다. **`qwen3.5:4b`를 제외한 주된 사유는 오답 3건보다 중국어 토큰 혼입입니다** — 그대로 고객에게 보낼 수 없는 출력이기 때문입니다.

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
| Python | 3.12.13 |
| 패키지 관리 | uv (`uv.lock` 고정) |
| `ollama` (Python 패키지) | 0.6.2 |
| `openai` (Python 패키지) | 3.8.0 |
| Ollama (서버) | 0.34.0 |
| GPU / VRAM | NVIDIA GeForce RTX 5060 Laptop GPU / 8,151 MiB |
| 시스템 RAM | 31.4 GB |

생성 설정은 모든 모델·모든 회차에 동일하게 적용했습니다.

**발제문은 생성 설정의 값을 지정하지 않습니다.** 요구는 "동일한 질문·지시문·대화 이력 처리·출력 한도·생성 설정을 적용하고, 차이가 있으면 명시" 와 "해당 조건을 결과 파일에 기록" 입니다. 아래 값은 **본 프로젝트가 정한 것**이며 근거는 다음과 같습니다.

`think=False` — 임의 선택이 아닙니다. Qwen3.5 계열은 thinking이 기본 활성이라 `num_ctx=4096`에서 기본값으로 실행하면 thinking이 컨텍스트를 모두 소진해 **빈 응답을 반환합니다**(2-8 참조). 두 모델의 비교 조건을 맞추기 위해 전 모델에 동일하게 비활성화했습니다.

`temperature=0.2` — **0을 쓰지 않은 이유가 있습니다.** 실측 결과 `temperature=0`에서는 같은 질문 2회의 응답이 완전히 동일했습니다.

| 설정 | 같은 질문 2회 실행 |
|---|---|
| `temperature=0` | 완전히 동일 (결정론적) |
| `temperature=0.2` | 응답이 갈림 |

[data/rubric.md](data/rubric.md)의 P4 판정 기준은 **"Q9·Q10 4회 중 항목 5에서 1점 이상 3회 이상"** 입니다. 회차마다 응답이 달라질 수 있다는 전제이며, `temperature=0`이면 4회가 전부 같아 `0/4` 또는 `4/4`만 나오므로 "3회 이상" 기준이 성립하지 않습니다. 채점 주의사항의 *"같은 질문의 2회 응답은 각각 채점하고 개별 점수를 모두 남깁니다"* 도 마찬가지입니다. 루브릭은 첫 호출 이전에 확정·커밋([`7800589`](https://github.com/jeongrolee-tech/local-llm-travel-assistant/commit/7800589))했으므로, 이와 정합하려면 `temperature > 0`이어야 합니다.

`seed` — 고정하지 않습니다. 고정하면 반복 회차가 결정론적이 되어 위와 같은 문제가 생깁니다.

**대가도 기록합니다.** `temperature > 0`이므로 1회 실행의 응답 차이에는 샘플링 변동이 섞입니다. 실제로 예비 실행에서 이를 설정 변경 효과로 오독한 사례가 있었고 재검증으로 바로잡았습니다(2-8). **질문당 2회를 수행하는 이유이기도 합니다.**

| 설정 | 값 |
|---|---|
| temperature | 0.2 (본 프로젝트가 정한 값 — 근거는 아래) |
| 출력 한도 | 미설정 (`num_predict` 기본값) |
| context_length | 4096 (`num_ctx`) |
| thinking | **비활성 (`think=False`)** — 전 모델 동일 |
| seed | 고정하지 않음 |
| 대화 이력 | 매 호출 초기화 (단발 질의) |

---

## 5. 실행 방법

```bash
# 1. 의존성 설치
uv sync

# 2. Ollama 모델 준비
ollama pull qwen3.5:9b      # 로컬 A
ollama pull gemma4:e4b      # 로컬 B

# 3. 로컬 실험 (워밍업 1회 후 본 실험 20회)
#    num_ctx=4096 / temperature=0.2 / think=False 고정
uv run python src/run_local.py --model qwen3.5:9b
uv run python src/run_local.py --model gemma4:e4b

# 4. Cloud 실험 (gpt-4o-mini, 공통 질문 5개 × 1회)
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

**로컬 본 실험 40회 완료** (2026-09-17). Cloud 비교는 3일차에 수행합니다.

| 모델 | 종합 점수 (100점 환산) | n | 성공/시도 | 평균 응답 시간 | 생성 속도 | VRAM | 필수 조건 |
|---|---|---|---|---|---|---|---|
| 로컬 A `qwen3.5:9b` | 88.1 | 20 | 20 / 20 | 1.56초 | 57.2 tok/s | 5,236 MiB | P1 ✅ P2 ✅ P3 ✅ **P4 ✅ (4/4)** |
| **로컬 B `gemma4:e4b`** | **92.9** | 20 | 20 / 20 | **0.91초** | **80.5 tok/s** | **3,077 MiB** | P1 ✅ P2 ✅ P3 ✅ **P4 ✅ (4/4)** |
| Cloud `gpt-4o-mini` | `{}` | `{}` | `{}` | `{}` | — | — | 비교 기준 (3일차) |

### 항목별 점수 (각 0~2점)

| 항목 | 로컬 A | 로컬 B | n |
|---|---|---|---|
| 1. 정확성 | 1.80 | **1.90** | 20 |
| 2. 핵심 정보 누락 | 1.50 | **1.55** | 20 |
| 3. 지시·형식 준수 | 1.85 | **1.90** | 20 |
| **4. 한국어 표현** | 1.65 | **1.95** | 20 |
| 5. 정보 부족 시 대응 | 2.00 | 2.00 | 4 (Q9·Q10) |

### 선정 판정 경로

**두 후보 모두 필수 통과 조건 4개를 충족했으므로** 선호 우선순위를 적용했습니다.

| 순위 | 지표 | 로컬 A | 로컬 B | 판정 |
|---|---|---|---|---|
| 1 | 종합 환산 점수 | 88.1 | 92.9 | 차이 **4.8점** — 루브릭의 "5점 이내면 동점" 에 해당해 **2순위로 이월** |
| 2 | **한국어 표현 점수** | 1.65 | **1.95** | 차이 **0.30점** — "0.2점 이내면 3순위" 기준을 넘어 **여기서 결판** |
| 3 | 전체 응답 시간 | 1.56초 | 0.91초 | (참고 — B 우세) |
| 4 | VRAM | 5,236 MiB | 3,077 MiB | (참고 — B 우세) |

**→ 로컬 B `gemma4:e4b` 선정.** 1순위에서 결판나지 않았고 2순위에서 갈렸습니다. 3·4순위도 같은 방향이지만 판정 근거는 2순위입니다.

### 점수를 가른 지점

**한국어 표현에서 로컬 A가 감점된 5개 회차입니다.**

| 회차 | 사유 | 항목4 |
|---|---|---|
| Q4 run2 | `이라는`**`项规定`**`가 있으나` — **중국어 혼입** | **0점** |
| Q2 run1·run2 | `([요금 기준](#2.-요금-기준))` 마크다운 링크 노출 — 그대로 발송 불가 | 1점 |
| Q6 run2 | `정확한 내용을 담당자 확인을 부탁드립니다` 비문 | 1점 |
| Q10 run1·run2 | `해당 내용은 오사카 … 약관에 명시되어 있습니다` — 베트남 내용이 있다는 오독 유발 | 1점 |

중국어 혼입은 예비 실행에서 `qwen3.5:4b`를 제외한 사유이기도 했습니다(2-8). **같은 계열의 9B 본 실험에서도 20회 중 1회 재현되었습니다.**

**문항별로는 로컬 A가 Q3에서 앞섰습니다.** A는 `712,000원(890,000 × 80%)`까지 계산했고, B는 `80%를 계산해야 합니다`로 고객에게 계산을 넘겼습니다. 예비 실행에서 관찰한 "A는 계산하고 B는 회피" 패턴이 본 실험에서도 나타났으나, 전체 점수를 뒤집지는 못했습니다.

회차별 점수와 점수 근거는 [results/scores.csv](results/scores.csv)에, 문항별 분석과 대표 실패 사례는 [docs/03_benchmark.md](docs/03_benchmark.md)에 있습니다.

워밍업은 모델당 1회로 [results/local_warmup.jsonl](results/local_warmup.jsonl)에 분리 저장했으며 위 집계에 포함하지 않았습니다. 호출 실패 0건이라 `results/errors.jsonl`은 생성되지 않았습니다.

**측정 조건 확인** — 두 모델 모두 `digest`·`quantization(Q4_K_M)`·`context_length(4096)`·생성 설정이 전 회차 동일하고, 40회 전부 **100% GPU 적재**에 `done_reason=stop`입니다. 생성 속도 계산 불가 회차는 0건입니다.

**P1 — 통과.** 두 모델 모두 질문 10개 전체에 응답을 반환했습니다 (20/20).

**P3 — 통과.** 입력 + 출력 최대값이 `qwen3.5:9b` 1,094 / 4,096 (27%), `gemma4:e4b` 1,204 / 4,096 (29%)입니다.

**평균 응답 시간과 생성 속도는 로컬 B가 앞섭니다.** 다만 이 값만으로 순위를 정하지 않습니다. 선호 우선순위는 종합 품질 점수 → 한국어 표현 점수 → 응답 시간 → VRAM 순이며, 품질 채점 결과가 나온 뒤 적용합니다.

Cloud는 질문 5개 × 1회, 로컬은 10개 × 2회로 **반복 수가 다릅니다.** 직접 순위 비교의 근거로 사용하지 않았습니다.

---

## 8. 검토했으나 채택하지 않은 개선안

예비 실행(2-8)에서 드러난 두 후보의 강점 차이를 근거로 두 가지 구성을 검토했습니다. 둘 다 본 실험에는 적용하지 않았으며, 사유를 아래에 기록합니다.

### 8-1. 질문 유형별 모델 라우팅

**착안점.** `qwen3.5:9b`는 금액을 실제로 계산한 유일한 후보였고(Q2 534,000원, Q3 712,000원), `gemma4:e4b`는 문서 독해와 경계 사례 처리(Q4·Q6·Q8)가 안정적이었습니다. 질문 유형에 따라 두 모델에 나눠 보내는 구성을 검토했습니다.

**실측 — 두 모델이 동시에 상주하지 않습니다.**

기본 설정과 `OLLAMA_MAX_LOADED_MODELS=2`로 재시작한 상태 **양쪽에서** 확인했습니다. 결과가 같습니다.

| 조합 | 실제 적재 합계 | 기본 설정 | `MAX_LOADED_MODELS=2` | 스케줄러 예측치 / 가용 |
|---|---|---|---|---|
| `qwen3.5:9b` + `gemma4:e4b` | 8,313 MiB | evict | **evict** | 9.4 GiB / 1.8 GiB |
| `qwen3.5:9b` + `gemma4:e2b` | 6,865 MiB | evict | **evict** | 6.8 GiB / 1.8 GiB |
| `qwen3.5:4b` + `gemma4:e2b` | 4,612 MiB | evict | **evict** | 6.8 GiB / 4.0 GiB |

**`OLLAMA_MAX_LOADED_MODELS`는 레버가 아니었습니다.** 2로 올려도 동일하게 evict됩니다. 실제 원인은 서버 로그에 있습니다.

```
msg="llama-server model predicted to exceed available memory, evicting"
    predicted="6.8 GiB" predicted_num_ctx=4096 num_batch=512 available="4.0 GiB"
```

**Ollama 스케줄러의 사전 메모리 예측치가 실제 적재량보다 훨씬 큽니다.** `gemma4:e2b`는 적재 후 `size_vram`이 **1,629 MiB**인데, 적재 전 예측은 **6.8 GiB**로 약 4배입니다. 예측에는 멀티모달 projector의 worst-case가 포함됩니다.

```
srv load_model: [mtmd] estimated worst-case memory usage of mmproj is 1152.07 MiB   # gemma4
srv load_model: [mtmd] estimated worst-case memory usage of mmproj is  986.67 MiB   # qwen3.5
```

후보 4개가 모두 멀티모달(vision/audio)이라 이 비용을 피할 수 없습니다. **8,151 MiB 단일 GPU에서는 설정 변경으로 해결되지 않습니다.**

> 검증은 임시 서버(`OLLAMA_MAX_LOADED_MODELS=2`)로 수행한 뒤 원래 구성으로 복구했습니다. 사용자 환경에 이 변수를 영구 등록하지 않았습니다.

**품질 이득의 상한 — 예비 실행 데이터로 계산.**

라우팅이 완벽하다고 가정하고(오라클 라우팅) 문항별로 두 모델 중 좋은 쪽을 취하면 상한이 나옵니다.

| | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 | Q9 | Q10 | ✅ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `qwen3.5:4b` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ◐ | ✅ | 6 |
| `gemma4:e2b` | ◐ | ◐ | ❌ | ◐ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | 5 |
| **오라클 합집합** | ◐ | ◐ | ✅ | ✅ | ✅ | ✅ | ✅ | **❌** | ✅ | ✅ | **7** |
| **`gemma4:e4b` 단독** | ◐ | ❌ | ◐ | ✅ | ✅ | ✅ | ✅ | **✅** | ✅ | ✅ | **7** |

**완벽하게 라우팅해도 단일 모델 `gemma4:e4b`와 같은 ✅7에 그칩니다.** Q8(경계 사례)은 4B 두 모델이 모두 실패해 라우팅으로 구제되지 않지만 `gemma4:e4b`는 단독으로 맞혔습니다. 실제 라우터는 오라클보다 나쁘므로 상한에도 못 미칩니다.

이는 **기존 예비 실행 데이터로 계산한 상한이며 라우팅을 실제로 실행해 측정한 값이 아닙니다.** 1회 실행 기준의 간이 판정이라는 한계도 그대로 적용됩니다.

**모델 전환 비용.**

| 모델 | 로딩 시간 (`load_duration`) | 생성 시간 (10문제 평균) |
|---|---|---|
| `qwen3.5:9b` | 6.0s | 1.65s |
| `gemma4:e4b` | 4.9s | 0.89s |
| `gemma4:e2b` | 4.5s | 0.55s |

**미채택 사유.** 단일 모델이 답변 1건을 1초 안팎에 반환하는데, 모델을 바꾸면 로딩에 5~6초가 추가됩니다. 본 Use Case는 상담 직원이 질문을 하나씩 처리하는 실시간 사용이므로 유형이 섞여 들어오면 매 전환마다 이 비용이 발생합니다. 질문을 유형별로 모아 배치 처리하면 전환이 1회로 줄지만, 그것은 본 Use Case의 사용 방식이 아닙니다.

**운영 권고.** 재검토하려면 아래 중 하나가 필요합니다. 설정 조정만으로는 되지 않습니다.

- **더 큰 VRAM** — 스케줄러 예측치 기준으로 두 모델 합계가 들어가야 하므로, 실제 적재량(8.3 GiB)이 아니라 **예측치 합계(약 15 GiB)**를 기준으로 잡아야 합니다.
- **텍스트 전용 모델** — 예측치의 상당 부분이 멀티모달 projector worst-case입니다. 본 Use Case는 텍스트만 다루므로, vision/audio가 없는 모델을 쓰면 예측치가 크게 줄어듭니다. 다만 현재 후보 4개는 모두 멀티모달입니다.

그리고 근거가 **예비 실행 1회**이므로, 본 실험 2회 결과에서 이 강점 분리가 재현되는지 먼저 확인해야 합니다.

### 8-2. 시스템 지시문 개선

**착안점.** 두 후보의 대표 실패가 모델 한계가 아니라 지시문에서 비롯됐을 가능성이 있습니다.

| 모델 | 실패 | 관찰 | 가설 |
|---|---|---|---|
| `qwen3.5:9b` | 과잉 회피 (Q4·Q6) | 문서에 명시된 내용인데 "확인이 어렵습니다"로 답함 | 규칙 1(근거 없으면 확인 불가 안내)이 과하게 작동 |
| `gemma4:e4b` | 계산 회피 (Q2·Q3) | 요금 구간은 맞게 식별하나 금액을 산출하지 않음 | 지시문에 "금액을 계산해 제시하라"는 요구가 없음 |

**미채택 사유.** [data/system_prompt.txt](data/system_prompt.txt)는 첫 호출 이전에 확정·커밋([`7800589`](https://github.com/jeongrolee-tech/local-llm-travel-assistant/commit/7800589))한 고정 지시문입니다. 실험 도중 변경하면 모델 간 비교 조건이 깨지고, 변경 전후 결과를 같은 실험으로 집계할 수 없습니다. **본 실험은 현재 지시문 그대로 수행합니다.**

이 개선안은 최종 보고서의 **"개선 필요 실패 사례"** 항목에 기록합니다. 지시문 수정 효과를 실제로 확인하려면 본 실험과 분리된 별도 실험으로 수행해야 합니다.

---

## 9. 한계

- 질문 10개, 반복 2회의 소규모 평가셋입니다.
- 단일 노트북 1대에서 측정한 값이므로 다른 장비의 절대 속도와 합산하지 않았습니다.
- `size_vram`은 측정 시점의 값이며 최대 VRAM 사용량이 아닙니다.
- `ollama ps`의 PROCESSOR는 CPU/GPU 적재 상태이며 GPU 이용률이 아닙니다.
- 후보 선별 예비 실행(2-8)은 모델당 1회이므로 P4 정식 판정(Q9·Q10 4회 기준)이 아닙니다. 선별 근거로만 사용했습니다.
- Cloud 비교 질문 5개를 로컬 예비 실행 결과를 본 뒤에 선정했습니다. 결과와 무관한 사전 규칙을 먼저 정하고 기계적으로 도출했으나, "결과를 보기 전 선정"이라는 원래 절차는 지키지 못했습니다. 선정 규칙과 사유는 [data/questions.json](data/questions.json)의 `cloud_subset`에 기록했습니다.
- 로컬 B(`gemma4:e4b`)는 thinking을 비활성화한 상태로만 측정했습니다. thinking 활성 시의 품질은 본 실험 범위 밖입니다.
- 다중 모델 동시 적재는 기본 설정과 `OLLAMA_MAX_LOADED_MODELS=2` 양쪽에서 확인했습니다. 그 외 설정(`OLLAMA_GPU_OVERHEAD`, `num_batch` 축소, 더 작은 `num_ctx`)으로 스케줄러 예측치를 낮출 수 있는지는 검증하지 않았습니다(8-1).
- 모델 크기와 품질의 관계는 후보 5개, 질문 10개, 1회 실행에서 관찰한 것입니다. 일반적인 결론으로 확장하지 않았습니다.
- 공개 Benchmark는 모델 카드 게시값을 옮긴 것이며 직접 재현하지 않았습니다. 평가 프로토콜이 모델마다 달라 순위 근거로 사용하지 않았습니다.
- 라우팅의 품질 이득은 예비 실행 데이터로 계산한 상한이며, 라우팅을 실제로 실행해 측정하지 않았습니다(8-1).
- CLI 실행 기록은 실행 가능 여부 확인용입니다. `ollama run` 기본 설정을 사용해 Python 경로와 생성 설정이 다르므로 측정값을 비교하지 않았습니다.
- 예비 실행은 질문당 1회이고 `temperature=0.2`이므로 응답 차이에 샘플링 변동이 섞여 있습니다. 실제로 이를 설정 변경 효과로 오독한 사례가 있었고 재검증으로 바로잡았습니다(2-8). 본 실험은 질문당 2회로 이 위험을 줄입니다.

---

## 10. 문서

| 문서 | 내용 |
|---|---|
| [docs/02_model_comparison.md](docs/02_model_comparison.md) | **Model Comparison Table (산출물 2번)** |
| [docs/cli/cli_session.txt](docs/cli/cli_session.txt) | CLI(`ollama run`) 실행 기록 |
| [docs/walkthrough.md](docs/walkthrough.md) | 수동 실행 워크스루 (발제문 STEP 4) |
| [docs/walkthrough_log.md](docs/walkthrough_log.md) | 수동 실행 기록 (환경·측정값·오류 증빙) |
| [docs/pilot/](docs/pilot/) | 후보 선별 예비 실행 원본 기록 (본 실험 아님) |
| [docs/03_benchmark.md](docs/03_benchmark.md) | 실험 결과와 품질 평가 |
| [docs/04_local_vs_cloud.md](docs/04_local_vs_cloud.md) | Local LLM vs Cloud API 비교 |
| [docs/05_selection_report.md](docs/05_selection_report.md) | 최종 선정 보고서 |

---

## 11. 저장소에 포함하지 않는 것

- 모델 가중치 파일
- API 키 및 `.env`
- 가상환경 디렉터리 전체
- 실제 고객 데이터 (본 실험은 가상 문서만 사용)
