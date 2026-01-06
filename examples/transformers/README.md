# A.X-K1 Transformers 추론 예제

[A.X-K1 모델](https://huggingface.co/skt/A.X-K1)을 HuggingFace Transformers로 분산 추론하는 예제입니다.

## 요구사항

- **하드웨어**: H100 GPU 8개 × 4노드 (총 32 GPUs)
- **Python**: 3.11+

## 설치

```bash
pip install torch transformers accelerate
```

## 파일 구성

| 파일 | 설명 |
|------|------|
| `inference.py` | 분산 추론 스크립트 |
| `inference.batch` | SLURM 배치 스크립트 |

## 실행 방법

### SLURM 환경

1. `inference.batch` 파일에서 가상환경 경로 설정:
   ```bash
   export VENV=<YOUR_VENV_PATH>
   ```

2. 배치 작업 제출:
   ```bash
   sbatch inference.batch
   ```

### 직접 실행 (torchrun)

```bash
torchrun --nnodes=4 --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    inference.py \
        --model_dir skt/A.X-K1 \
        --prompt "MoE에 대해서 설명해 주세요" \
        --max_new_tokens 64
```

## 인자

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--model_dir` | 모델 경로 또는 HuggingFace ID | (필수) |
| `--prompt` | 입력 프롬프트 | (필수) |
| `--max_new_tokens` | 최대 생성 토큰 수 | 4096 |
| `--disable_kv_cache` | KV 캐시 비활성화 | False |

## 참고사항

- 모델은 Expert Parallelism (EP)으로 분산되며, `ep_size`는 자동으로 `world_size`(32)로 설정됩니다.
- `enable_thinking=True` 옵션으로 reasoning 모드가 활성화됩니다.
