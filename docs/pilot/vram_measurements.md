# VRAM 실측 기록

측정 일자 2026-09-16. GPU **NVIDIA GeForce RTX 5060 Laptop GPU, 8,151 MiB** / RAM 31.4 GB / Ollama 0.34.0.

## 측정 방법

각 모델을 `num_predict=1`로 1회 로드한 직후 Ollama `/api/ps`의 `size_vram`(GPU 적재분)과 `size`(전체 적재분)를 읽습니다. 측정 후 `keep_alive=0`으로 내린 뒤 다음 모델을 올립니다. 스크립트는 [`measure_vram.sh`](measure_vram.sh)입니다.

> `size_vram`은 **측정 시점의 값이며 최대 VRAM 사용량이 아닙니다.**
> `size == size_vram`이면 100% GPU 적재, 작으면 차액만큼 CPU로 분할된 상태입니다.
> 이 값은 CPU/GPU **적재 상태**이며 GPU 이용률이 아닙니다.

## 모델별 (num_ctx = 4096)

| 모델 | 디스크 | `size_vram` | `size` | 적재 |
|---|---|---|---|---|
| `qwen3.5:4b` | 3.4 GB | 2,983 MiB | 2,983 MiB | 100% GPU |
| `qwen3.5:9b` | 6.6 GB | 5,236 MiB | 5,236 MiB | 100% GPU |
| `gemma4:e2b` | 7.2 GB | 1,629 MiB | 1,629 MiB | 100% GPU |
| `gemma4:e4b` | 9.6 GB | 3,077 MiB | 3,077 MiB | 100% GPU |
| `exaone3.5:7.8b` | 4.8 GB | 4,945 MiB | 4,945 MiB | 100% GPU |
| Kanana 2 3B (Q4_K_M) | 2.2 GB | 2,462 MiB | 2,462 MiB | 100% GPU |

**디스크 크기와 VRAM은 같지 않습니다.** `gemma4:e2b`는 디스크 7.2 GB인데 GPU 적재는 1,629 MiB입니다. Per-Layer Embeddings 테이블이 GPU에 상주하지 않기 때문입니다. 디스크 크기 / 시스템 RAM / VRAM은 서로 다른 값이므로 구분해 기록합니다.

## context_length에 따른 변화

`num_ctx`를 키우면 KV 캐시가 늘어 8,151 MiB 한계를 넘는 시점부터 CPU로 분할됩니다.

| 모델 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|
| `qwen3.5:9b` | **5,236 / 5,236 MiB (100% GPU)** | 5,251 / 5,987 MiB (88% GPU) | 5,244 / 6,251 MiB (84% GPU) | 5,259 / 6,907 MiB (76% GPU) |
| `qwen3.5:4b` | 2,983 / 2,983 MiB (100%) | 3,187 / 3,187 MiB (100%) | 3,459 / 3,459 MiB (100%) | 4,003 / 4,003 MiB (100%) |
| Kanana 2 3B | 2,462 / 2,462 MiB (100%) | 3,052 / 3,052 MiB (100%) | 4,092 / 4,092 MiB (100%) | 5,927 / 6,551 MiB (90% GPU) |

`qwen3.5:9b`는 **`num_ctx=4096`에서만 100% GPU 적재**됩니다. 8192부터 CPU 분할이 시작됩니다.

## 실험 context_length를 4096으로 고정한 근거

실측 입력 토큰은 시스템 지시문 + 정책 문서 + 질문 합계 **977~987 토큰**(Qwen3.5 기준), **807~818 토큰**(Kanana 기준)입니다. `think=False`에서 출력은 30~140 토큰이므로 4096 안에 충분히 수용됩니다(P3 충족).

단 **thinking을 켜면 초과합니다.** `qwen3.5:9b`를 `num_ctx=4096` / think on으로 실행하면 thinking이 컨텍스트를 모두 소진합니다.

| 설정 | 응답 시간 | 토큰 | 적재 | 결과 |
|---|---|---|---|---|
| `num_ctx=4096`, think **on** | 62.2s | 979 + 3,119 = **4,096 소진** | 100% GPU | **빈 응답** (P1 실패) |
| `num_ctx=8192`, think on | 77.2s | 977 + 3,458 = 4,435 | 88% GPU | 응답 정상 |
| `num_ctx=4096`, think **off** | **7.5s** | 979 + 57 = 1,036 | **100% GPU** | **응답 정상** |

따라서 본 실험은 `num_ctx=4096` + `think=False`를 전 모델에 동일 적용합니다.

## 다중 모델 동시 적재

두 모델을 연달아 적재하고 `ollama ps`로 잔류 여부를 확인했습니다 (각 `num_ctx=4096`, `keep_alive=5m`).

| 조합 | 합계 VRAM | 여유 (8,151 MiB 기준) | 결과 |
|---|---|---|---|
| `qwen3.5:9b` + `gemma4:e4b` | 8,313 MiB | 초과 | 두 번째 적재 시 첫 번째 evict |
| `qwen3.5:9b` + `gemma4:e2b` | 6,865 MiB | 1,286 MiB 여유 | 동일하게 evict |
| `qwen3.5:4b` + `gemma4:e2b` | 4,612 MiB | 3,539 MiB 여유 | 동일하게 evict |

**VRAM이 남아도 evict됩니다.** 절반 조금 넘게 쓰는 조합에서도 같은 결과이므로, 원인은 VRAM 용량이 아니라 Ollama의 적재 정책입니다.

```
OLLAMA_MAX_LOADED_MODELS = (미설정)
OLLAMA_NUM_PARALLEL      = (미설정)
OLLAMA_GPU_OVERHEAD      = (미설정)
```

이 값들을 변경하려면 Ollama 서버 재시작이 필요하며, **본 실험에서는 변경해 검증하지 않았습니다.**

### 모델 전환 비용

| 모델 | 로딩 시간 (`load_duration`) | 생성 시간 (10문제 평균) |
|---|---|---|
| `qwen3.5:9b` | 6.0s | 1.65s |
| `gemma4:e4b` | 4.9s | 0.89s |
| `gemma4:e2b` | 4.5s | 0.55s |
| Kanana 2 3B | 2.8s | 0.19s |

단일 모델 응답이 1초 안팎인데 전환에 5~6초가 듭니다. 질문 유형별 라우팅을 채택하지 않은 근거입니다 (README 8-1).
