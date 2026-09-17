# 후보 선별 예비 실행 기록

> **본 실험이 아닙니다.** 로컬 A·B를 고르기 위한 STEP 3 후보 선별용이며, **모델 5개 × 질문 10개 × 1회 = 50회**입니다.
> 필수 본 실험(**최종 후보 2개 × 10문제 × 2회 = 40회**)과 분리되며, 본 실험 집계에 포함하지 않습니다.
> 판정 요약과 결론은 [README 2-8](../../README.md)에 있습니다.

## 실행 조건 (전 모델 동일)

| 항목 | 값 |
|---|---|
| `num_ctx` | 4096 |
| `temperature` | 0.2 |
| `think` | `False` |
| 대화 이력 | 매 호출 초기화 (단발 질의) |
| 출력 한도 | 미설정 (`num_predict` 기본값) |
| 워밍업 | 모델당 1회 선행, 집계 분리 |
| 반복 | 질문당 1회 |
| 측정 일자 | 2026-09-16 |
| 실행 환경 | Windows 11 Pro 10.0.26200 / RTX 5060 Laptop GPU 8,151 MiB / RAM 31.4 GB / Ollama 0.34.0 |

입력은 [data/system_prompt.txt](../../data/system_prompt.txt) + [data/policy.md](../../data/policy.md) + [data/questions.json](../../data/questions.json)의 질문 10개입니다.

## 파일

| 파일 | 내용 |
|---|---|
| `01_qwen3.5-9b_kanana-raw-gguf.txt` / `.jsonl` | `qwen3.5:9b`, Kanana 2 3B **변환본 원본 상태**(채팅 템플릿 누락) |
| `02_kanana-template-restored.txt` / `.jsonl` | Kanana 2 3B **템플릿 복구 후** 재측정 |
| `03_gemma4-e2b-e4b_qwen3.5-4b.txt` / `.jsonl` | `gemma4:e2b`, `gemma4:e4b`, `qwen3.5:4b` |
| `run_pilot.py` | 예비 실행 스크립트 (재현용) |
| `Modelfile.kanana2-3b-chatml` | Kanana 변환본 채팅 템플릿 복구용 Modelfile |
| `measure_vram.sh` | 모델별 `size_vram` 측정 스크립트 |
| `vram_measurements.md` | VRAM 실측 기록 |

`.jsonl`은 회차별 레코드입니다 — 모델 태그, 질문 ID, 유형, 응답 시간, 입력·출력 토큰 수, 생성 속도, 응답 전문.

## 재현

```bash
# 예비 실행
uv run python docs/pilot/run_pilot.py qwen3.5:9b gemma4:e4b gemma4:e2b qwen3.5:4b

# Kanana 변환본 템플릿 복구 후 실행
ollama create kanana2-3b-chatml:q4km -f docs/pilot/Modelfile.kanana2-3b-chatml
uv run python docs/pilot/run_pilot.py kanana2-3b-chatml:q4km

# VRAM 측정
bash docs/pilot/measure_vram.sh
```

## 주의

- ✅/◐/❌ 간이 판정은 후보 선별용이며 [data/rubric.md](../../data/rubric.md)의 5항목 × 0~2점 정식 채점이 **아닙니다.**
- P4 정식 판정은 Q9·Q10 **4회** 기준입니다. 이 기록은 1회이므로 정식 판정이 아니라 선별 근거입니다.
- Kanana 2 3B 결과는 **커뮤니티 GGUF 변환본 기준**이며, 원본 `kakaocorp/kanana-2-3b-instruct`의 성능과 동일하다고 단정하지 않습니다.
- `exaone3.5:7.8b`는 License(EXAONE 1.1 - NC)가 P2를 충족하지 못해 **품질 측정을 수행하지 않았습니다.** VRAM만 기록했습니다.
