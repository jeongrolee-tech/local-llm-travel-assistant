# Model Comparison Table

발제문 산출물 2번입니다. 후보 6개의 Model Card 조사 결과와 실행 환경 실측값을 모았습니다.

- 조사·측정 기준일: 2026-09-16 ~ 2026-09-17
- 측정 환경: Windows 11 Pro 10.0.26200 / NVIDIA GeForce RTX 5060 Laptop GPU 8,151 MiB / RAM 31.4 GB / Ollama 0.34.0
- 선정 결과와 실험 설계는 [README](../README.md), 예비 실행 원본은 [pilot/](pilot/), CLI 실행 기록은 [cli/](cli/)에 있습니다.

---

## 1. 전체 비교표

| 항목 | `qwen3.5:9b` (로컬 A) | `gemma4:e4b` (로컬 B) | `qwen3.5:4b` | `gemma4:e2b` | `exaone3.5:7.8b` | Kanana 2 3B |
|---|---|---|---|---|---|---|
| **Model Name** | Qwen3.5-9B | Gemma 4 E4B-it | Qwen3.5-4B | Gemma 4 E2B-it | EXAONE-3.5-7.8B-Instruct | kanana-2-3b-instruct |
| **배포처** | Alibaba / Qwen | Google DeepMind | Alibaba / Qwen | Google DeepMind | LG AI Research | Kakao |
| **Parameter Size** | 9.7B | 유효 4.5B / 전체 8.0B | 4.7B | 유효 2.3B / 전체 5.1B | 7.8B | 3.51B |
| **Architecture** | `qwen35` — Gated DeltaNet + sparse MoE, 32 layers | `gemma4` — Dense + Per-Layer Embeddings, SWA + global attention, 42 layers | `qwen35` (동일 계열), 32 layers | `gemma4` (동일 계열), 35 layers | `exaone` — GQA 32Q/8KV, 32 layers | `qwen3` (`Qwen3ForCausalLM`), 32 layers, GQA 32Q/8KV |
| **Language** | 201개 언어·방언 | 기본 35개+ / 사전학습 140개+ | 201개 언어·방언 | 기본 35개+ / 사전학습 140개+ | 한국어·영어 | 한국어·영어 |
| **Context Length (카드 최대)** | 262,144 (YaRN ~1,010,000) | 131,072 | 262,144 (YaRN ~1,010,000) | 131,072 | 32,768 | 32,768 |
| **Context Length (실험 설정)** | **4,096** | **4,096** | 4,096 | 4,096 | 미실행 | 4,096 |
| **모달리티** | Text / Image / Video | Text / Image / Audio | Text / Image / Video | Text / Image / Audio | Text | Text |
| **Quantization** | Q4_K_M | Q4_K_M | Q4_K_M | Q4_K_M | Q4_K_M | Q4_K_M (GGUF 메타데이터 미기록) |
| **VRAM (실측, ctx 4096)** | 5,236 MiB | 3,077 MiB | 2,983 MiB | 1,629 MiB | 4,945 MiB | 2,462 MiB |
| **디스크 크기** | 6.6 GB | 9.6 GB | 3.4 GB | 7.2 GB | 4.8 GB | 2.2 GB |
| **License** | Apache 2.0 | Apache 2.0 (+ Prohibited Use Policy) | Apache 2.0 | Apache 2.0 (+ PUP) | **EXAONE 1.1 - NC** | Kanana Open License |
| **P2 (상업적 사용)** | 통과 | 통과 | 통과 | 통과 | **실패** | 통과 (조건부) |
| **선정 결과** | **로컬 A** | **로컬 B** | 제외 | 예비 | 제외 (License) | 제외 (P4) |

**주요 특징과 선정·탈락 이유**

| 모델 | 주요 특징 | 선정 / 탈락 이유 |
|---|---|---|
| `qwen3.5:9b` | 하이브리드 MoE, 최장 Context, 201개 언어, thinking 기본 활성 | **로컬 A.** 예비 실행에서 금액을 실제로 계산한 유일한 후보(534,000원·712,000원). 문서에 명시된 내용을 과잉 회피하는 실패 유형. |
| `gemma4:e4b` | Per-Layer Embeddings로 디스크 대비 VRAM이 작음, 오디오 입력 지원 | **로컬 B.** 문서 독해와 경계 사례 처리가 가장 안정적. 금액 계산을 회피하는 실패 유형. 로컬 A와 실패 방식이 달라 비교 축이 성립. |
| `qwen3.5:4b` | 로컬 A와 같은 계열의 소형 | 제외. 예비 실행 Q1 응답에 **중국어 토큰 혼입**(`出发`) — 루브릭 항목 4 "외국어 혼입" 0점 기준. 중요 가치 1순위(한국어 표현)와 직접 충돌. |
| `gemma4:e2b` | 후보 중 VRAM 최소(1,629 MiB), 응답 최속(평균 0.55초) | 예비. 정답 후 "확인이 어렵습니다"를 덧붙이는 자기모순 패턴. 로컬 B가 같은 계열에서 더 안정적이라 후순위. |
| `exaone3.5:7.8b` | 한국어·영어 2개국어 특화, KoMT-Bench 7.96 | **탈락 — P2 실패.** License 1.1-NC가 상업적 사용을 금지. 품질 측정을 수행하지 않았다. |
| Kanana 2 3B | 한국어 특화, 한국어 토크나이저 효율 최고, 응답 최속(0.19초) | **탈락 — P4 미달.** 예비 실행 Q9·Q10에서 문서에 없는 내용을 생성("3천만원", "90일 비자 면제"). `temperature=0` 재검증에서도 동일하게 재현되어 모델 원인으로 확정. |

---

## 2. 실행 식별값 — Ollama 태그와 원본 모델의 대응

| 실행 태그 | digest | 원본 모델 (Hugging Face) | 대응 관계 |
|---|---|---|---|
| `qwen3.5:9b` | `6488c96fa5fa` | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | Ollama 공식 라이브러리 배포본 |
| `gemma4:e4b` | `c6eb396dbd59` | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | Ollama 공식 라이브러리 배포본 (instruct 튜닝본) |
| `qwen3.5:4b` | `2a654d98e6fb` | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Ollama 공식 라이브러리 배포본 |
| `gemma4:e2b` | `7fbdbf8f5e45` | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | Ollama 공식 라이브러리 배포본 (instruct 튜닝본) |
| `exaone3.5:7.8b` | `c7c4e3d1ca22` | [LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct) | Ollama 공식 라이브러리 배포본 |
| `hf.co/mradermacher/kanana-2-3b-instruct-GGUF:Q4_K_M` | `d5a16a4a92bc` | [kakaocorp/kanana-2-3b-instruct](https://huggingface.co/kakaocorp/kanana-2-3b-instruct) | **커뮤니티 GGUF 변환본.** 카카오가 공식 GGUF를 배포하지 않음 |
| `kanana2-3b-chatml:q4km` | `ab937df3c73d` | 위 변환본 | 템플릿 누락으로 오판해 재빌드한 것. **재검증 결과 무효과**로 확인되어 사용하지 않음 ([Modelfile](pilot/Modelfile.kanana2-3b-chatml), [README 2-8](../README.md)) |

**다운로드 파일 크기 / 시스템 RAM / VRAM은 서로 다른 값입니다.** 특히 `gemma4:e2b`는 디스크 7.2 GB인데 VRAM은 1,629 MiB입니다. Per-Layer Embeddings 테이블이 GPU에 상주하지 않기 때문입니다. 자세한 측정은 [pilot/vram_measurements.md](pilot/vram_measurements.md)에 있습니다.

---

## 3. Tokenizer

| 모델 | Vocab Size | 동일 입력 프롬프트 토큰 (실측) | Kanana 대비 |
|---|---|---|---|
| Kanana 2 3B | 128,256 | **811.2** | 기준 |
| `qwen3.5:9b` / `qwen3.5:4b` | 248,320 | 979.6 | **+20.8%** |
| `gemma4:e4b` / `gemma4:e2b` | 262,144 | 1,071.6 | **+32.1%** |
| `exaone3.5:7.8b` | 102,400 | 미측정 (P2 실패로 미실행) | — |

측정 입력은 모든 모델에 동일합니다 — `data/system_prompt.txt` + `data/policy.md` + 질문 10개. 위 값은 예비 실행 10회의 평균이며 원본은 [pilot/](pilot/)의 `.jsonl`에 회차별로 있습니다.

**한국어 특화 토크나이저의 효율이 실제로 확인됩니다.** 같은 한국어 문서를 Kanana는 811 토큰, Gemma 4는 1,072 토큰으로 표현합니다. 32% 차이는 Context 여유와 입력 비용에 직접 영향을 줍니다. 다만 **본 실험에서 이 이점이 품질로 이어지지는 않았습니다** — Kanana는 P4에서 탈락했습니다.

---

## 4. Chat Template

Ollama 0.34는 모델에 따라 세 가지 방식으로 대화 형식을 적용합니다 — 내장 렌더러, Modelfile의 Go 템플릿, GGUF에 내장된 Jinja 템플릿입니다.

| 모델 | 적용 방식 | `ollama show --modelfile` 출력 |
|---|---|---|
| `qwen3.5:9b` / `qwen3.5:4b` | **내장 렌더러** | `TEMPLATE {{ .Prompt }}` + `RENDERER qwen3.5` + `PARSER qwen3.5` |
| `gemma4:e4b` / `gemma4:e2b` | **내장 렌더러** | `TEMPLATE {{ .Prompt }}` + `RENDERER gemma4` + `PARSER gemma4` |
| `exaone3.5:7.8b` | Go 템플릿 | `TEMPLATE {{- range $i, $_ := .Messages }}...` + `PARAMETER stop [\|endofturn\|]` + `SYSTEM` |
| `hf.co/mradermacher/kanana-2-3b-instruct-GGUF:Q4_K_M` | **GGUF 내장 Jinja** | `TEMPLATE {{ .Prompt }}` — 그러나 `--template`은 공식 Jinja **10,725 bytes** 출력 |

**`ollama show --modelfile`의 `TEMPLATE` 줄로는 템플릿 유무를 판단할 수 없습니다.** 위 네 모델이 모두 `TEMPLATE {{ .Prompt }}`를 출력하지만, 실제 대화 형식은 Qwen3.5·Gemma 4의 경우 `RENDERER`/`PARSER`가, Kanana 변환본의 경우 **GGUF에 내장된 공식 Jinja 템플릿**이 처리합니다. 확인하려면 `ollama show --template`을 사용해야 합니다.

Kanana 변환본의 템플릿이 누락되었다고 오판해 ChatML로 재빌드한 적이 있으나, 재검증 결과 **프롬프트 토큰 수와 출력이 모두 동일해 무효과**임을 확인했습니다. 오판과 정정 과정은 [README 2-8](../README.md)에 기록했습니다.

---

## 5. 공개 Benchmark

모델 카드에 게시된 점수입니다. **직접 측정한 값이 아닙니다.**

| 벤치마크 | `qwen3.5:9b` | `gemma4:e4b` | `qwen3.5:4b` | `gemma4:e2b` | `exaone3.5:7.8b` | Kanana 2 3B |
|---|---|---|---|---|---|---|
| MMLU-Pro | **82.5** | 69.4 | 79.1 | 60.0 | 46.24 | — |
| MMLU (CoT) | — | — | — | — | — | 61.09 |
| GPQA Diamond | 81.7 | 58.6 | 76.2 | 43.4 | — | — |
| MMMLU (다국어) | 81.2 | 76.6 | 76.1 | 67.4 | — | — |
| IFEval | 91.5 | — | 89.8 | — | 78.9 | 80.96 |
| MT-Bench | — | — | — | — | 8.29 | 7.15 |
| **KMMLU (CoT)** | 미공개 | 미공개 | 미공개 | 미공개 | 미공개 | **43.32** |
| **KoMT-Bench** | 미공개 | 미공개 | 미공개 | 미공개 | **7.96** | 6.92 |
| **HAE-RAE Bench** | 미공개 | 미공개 | 미공개 | 미공개 | 미공개 | 43.75 |
| LogicKor | — | — | — | — | 9.08 | — |

### 이 표를 해석할 때의 제약

**서로 다른 벤치마크는 직접 비교할 수 없습니다.** 모델마다 공개한 항목과 평가 프로토콜(few-shot 수, CoT 사용 여부, 채점 방식)이 다릅니다. MMLU-Pro처럼 항목명이 같아도 측정 조건이 같다는 보장은 없습니다.

**Qwen3.5 점수는 thinking 모드 기준일 가능성이 높습니다.** 모델 카드가 thinking 기본 활성을 명시합니다. 본 실험은 `think=False`로 수행하므로 (README 4절) 이 점수를 본 실험 성능의 예측값으로 쓸 수 없습니다.

**한국어 벤치마크가 결정적으로 부족합니다.** KMMLU·KoMT-Bench·HAE-RAE를 공개한 것은 한국어 특화 모델인 EXAONE 3.5와 Kanana 2뿐이고, 정작 로컬 A·B로 선정한 Qwen3.5와 Gemma 4는 **한국어 단독 벤치마크를 공개하지 않았습니다.** 다국어 종합 지표(MMMLU)만 있습니다.

→ **공개 벤치마크만으로는 이 Use Case(한국어 여행 상담)의 적합성을 판단할 수 없습니다.** 고정 질문 10개로 자체 측정을 수행하는 근거가 여기에 있습니다. 실측 결과는 [03_benchmark.md](03_benchmark.md)에 기록합니다.

---

## 6. License

| 모델 | License | 상업적 사용 | 배포 시 의무 | P2 |
|---|---|---|---|---|
| Qwen3.5 (9B / 4B) | Apache License 2.0 | 가능 | 라이선스 사본 첨부, 변경사항 표시 | 통과 |
| Gemma 4 (E4B / E2B) | Apache License 2.0 | 가능 | Apache 2.0 의무 + Google **Prohibited Use Policy** 준수 | 통과 |
| Kanana 2 3B | Kanana Open License Agreement (2025-07-17) | 조건부 가능 | **"Powered by Kanana" 표시** + Notice 파일 포함 | 통과 |
| EXAONE 3.5 7.8B | EXAONE AI Model License Agreement 1.1 - **NC** | **불가 (연구 목적 한정)** | 라이선스 사본 첨부, 파생 모델명에 "EXAONE" 접두 | **실패** |

근거 조항과 판단 과정은 [README 2-3](../README.md)에 있습니다.

---

## 7. 출처

| 대상 | 링크 |
|---|---|
| Qwen3.5-9B Model Card | https://huggingface.co/Qwen/Qwen3.5-9B |
| Qwen3.5-4B Model Card | https://huggingface.co/Qwen/Qwen3.5-4B |
| Qwen3.5 License (Apache 2.0) | https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE |
| Gemma 4 E4B Model Card | https://huggingface.co/google/gemma-4-E4B-it |
| Gemma 4 E2B Model Card | https://huggingface.co/google/gemma-4-E2B-it |
| Gemma 4 License | https://ai.google.dev/gemma/docs/gemma_4_license |
| Gemma 4 Prohibited Use Policy | https://ai.google.dev/gemma/prohibited_use_policy |
| Gemma 4 Technical Report | https://arxiv.org/abs/2607.02770 |
| EXAONE 3.5 Model Card | https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct |
| EXAONE 3.5 License (1.1 - NC) | https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct/blob/main/LICENSE |
| Kanana 2 3B Instruct Model Card | https://huggingface.co/kakaocorp/kanana-2-3b-instruct |
| Kanana Open License | https://huggingface.co/kakaocorp/kanana-2-3b-instruct/blob/main/LICENSE |
| Kanana 2 3B GGUF 변환본 (실사용) | https://huggingface.co/mradermacher/kanana-2-3b-instruct-GGUF |
| gpt-4o-mini (Cloud 비교) | https://developers.openai.com/api/docs/models/gpt-4o-mini |
| OpenAI API 가격 | https://developers.openai.com/api/docs/pricing |
