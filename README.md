<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"
            srcset="./assets/A.X_K1_BI_Negative.png">
    <img src="./assets/A.X_K1_BI_Primary.png" alt="A.X Logo" width="300">
  </picture>
</div>
<br/>
<p align="center">
  <a href="https://huggingface.co/skt/A.X-K1">🤗 Models</a> |
  <a href="https://github.com/SKT-AI/A.X-K1">🖥️ Github</a> |
  <a href="https://github.com/SKT-AI/A.X-K1/blob/main/A_X_Tech_Report.pdf">📄 Technical Report</a>
</p>

## Model Summary

**A.X K1** is a large-scale Mixture-of-Experts (MoE) language model designed for efficient high-capacity reasoning and instruction following, and was trained from scratch without initializing from an existing pretrained model.
The model contains **519 billion total parameters**, with **33 billion active parameters**, enabling strong performance while maintaining practical inference efficiency.

This hybrid design allows the user to choose between in-depth reasoning and response latency depending on task requirements.

## Key Features

- **Large-Scale Sparse MoE (519B / 33B Active)**
  Employs a high-capacity Mixture-of-Experts architecture that activates only a small subset of experts per token, enabling strong reasoning performance with practical inference efficiency.

- **Hybrid Reasoning Control (Think / Non-Think)**
  Supports user-controllable reasoning depth, allowing explicit multi-step reasoning or concise low-latency responses within a single unified model.

- **Tokenizer Optimized for Multilingual and Code Data**
  Uses a large-vocabulary BBPE-based tokenizer optimized for token efficiency across five languages (English, Korean, Chinese, Japanese, and Spanish), with a strong emphasis on source code, structured text, and programming-related patterns.

- **Stability-Oriented Architecture for Large-Scale MoE**
  Incorporates RMSNorm both before and after MLP (MoE) blocks, improving training stability and robustness in sparse, long-context settings.

## Model Details

- **Architecture:** Decoder-only Transformer with Mixture-of-Experts (MoE)
- **Total parameters:** 519B (192 experts + 1 shared expert)
- **Active parameters:** 33B per token (8 experts + 1 shared expert)
- **Number of layers:** 61 (1 dense + 60 MoE)
- **Number of attention heads:** 64
- **Intermediate size:** 7168
- **Expert intermediate size:** 2048
- **Normalization:** RMSNorm applied both before and after the MLP block
- **Attention:** Multi-Latent Attention (MLA)
- **Vocab size:** 163,840
- **Context length:** 131,072 tokens

## Architecture Highlights

### Mixture-of-Experts Design

A.X K1 follows a sparse Mixture-of-Experts architecture in which only a subset of experts is activated per token.
This design substantially increases model capacity while keeping the computational cost comparable to dense models with much smaller parameter counts.

From a scalability and efficiency perspective, MoE architectures enable model capacity to grow primarily by adding experts, with substantially slower growth in compute compared to dense models.
Expert parallelism allows experts to be distributed across devices, supporting large-scale training and serving without activating all parameters on every forward pass.
Recent MoE scaling-law studies provide guidance for selecting the number of experts and activation ratios under fixed compute and memory budgets.

### Hybrid Reasoning Fusion (Think / Non-Think)

A.X K1 uses a single model to generate responses where reasoning before the answer can be enabled or disabled depending on usage requirements.
This design supports controlled trade-offs between reasoning depth and response latency.

- **Think mode:**
  Generates reasoning steps before producing the answer for complex problem solving and multi-step inferences.
- **Non-Think mode:**
  Generates concise, direct responses optimized for low-latency usage.

### Post-MLP RMSNorm

A.X K1 incorporates an additional RMSNorm applied after the MLP (MoE) block in each Transformer layer.
This design choice improves training stability in large-scale sparse MoE settings and enhances robustness for both reasoning-intensive and long-context generations.

### Multi-Token Prediction (MTP)

During training, A.X K1 employs a multi-token prediction objective in which the model predicts one future token beyond the standard next-token objective from a single forward pass, serving as an auxiliary signal that helps stabilize training for large-scale models. At inference time, MTP does not modify the standard autoregressive decoding process, but provides benefits for speculative decoding, enabling higher inference throughput when used with compatible serving frameworks.

## Evaluation Results

### Thinking Mode

| Domain | Benchmark | Lang. | A.X K1 (519B-A33B) | DeepSeek-V3.1 (685B-A37B) | GLM-4.6 (357B-A32B) |
| --- | --- | --- | ---: | ---: | ---: |
| Knowledge | KMMLU | Korean | 80.2 | 76.5 | 79.9 |
| | KMMLU-Redux | Korean | 77.9 | 75.9 | 78.2 |
| | KMMLU-Pro | Korean | 68.1 | 71.4 | 71.8 |
| | CLIcK | Korean | 84.9 | 84.5 | 84.9 |
| | KoBALT | Korean | 48.8 | 59.7 | 59.2 |
| | MMLU-Pro | English | 81.5 | 85.1 | 82.9 |
| | GPQA-Diamond | English | 74.0 | 77.9 | 78.0 |
| Instruction Following | IFBench (prompt-loose) | English | 64.7 | 41.5 | 43.4 |
| | IFEval (prompt-strict) | English | 80.4 | 84.4 | 86.1 |
| | IFEval-ko (prompt-strict) | Korean | 81.0 | 79.2 | 85.8 |
| Math | AIME25 | English | 89.8 | 88.4 | 86.0 |
| | AIME25-ko | Korean | 80.4 | 81.3 | 80.4 |
| | HRM8K | Korean | 84.3 | 84.3 | 83.9 |
| Code | LiveCodeBench v6 (25.2.1–25.5.1) | English | 75.8 | 69.5 | 76.0 |
| | LiveCodeBench-ko | Korean | 73.1 | 66.2 | 55.9 |
| | HumanEval+ | English | 87.2 | 86.0 | 83.5 |
| | HumanEval+ ko | Korean | 90.2 | 93.9 | 86.0 |
| | MBPP+ | English | 93.0 | 99.2 | 98.9 |
| | SciCode | English | 32.4 | 39.1 | 38.4 |
| Long Context | AA-LCR | English | 36.0 | 53.3 | 54.3 |
| | Humanity’s Last Exam | English | 7.2 | 13.0 | 13.3 |
| Agent | τ² Telecom | English | 58.1 | 37.4 | 70.5 |

### Non-Thinking Mode

| Domain | Benchmark | A.X K1 (519B-A33B) | DeepSeek-V3.1 (685B-A37B) | GLM-4.6 (357B-A32B) |
| --- | --- | ---: | ---: | ---: |
| Knowledge | KMMLU | 73.0 | 78.7 | 77.7 |
| | KMMLU-Redux | 68.3 | 75.9 | 73.4 |
| | KMMLU-Pro | 60.3 | 67.9 | 68.2 |
| | CLIcK | 77.2 | 80.9 | 77.9 |
| Instruction Following | IFBench | 44.3 | 37.8 | 36.7 |
| | IFEval | 78.6 | 82.7 | 87.2 |
| Code | HumanEval+ | 79.9 | 87.8 | 89.0 |
| | HumanEval+ ko | 75.6 | 86.6 | 92.1 |
| | MBPP+ | 85.7 | 92.6 | 94.2 |

## Usage

### Transformers

For users who wish to run direct inference with Hugging Face Transformers, please refer to the example scripts and instructions in [this repository](./examples/transformers/README.md).

### vLLM

We provide an initial **vLLM** integration for A.X K1 in the following repository:

- <https://github.com/fort726/vllm/tree/add-ax-k1-model>

### SGLang

We provide an initial **SGLang** integration for A.X K1 in the following repository:

- <https://github.com/fort726/sglang/tree/add-ax-k1-model>

This setup has been validated in a multi-node, tensor-parallel configuration with long-context support.
Below is an example launch configuration used for evaluation.

#### Head node

```bash
python -m sglang.launch_server \
  --model-path $MODEL \
  --nccl-init-addr ${HEAD_NODE_IPADDRESS}:25000 \
  --nnodes 4 \
  --node-rank 0 \
  --tp-size 32 \
  --context-length 131072 \
  --max-running-requests 4 \
  --cuda-graph-max-bs 4
```

#### Worker nodes

```bash
python -m sglang.launch_server \
  --model-path $MODEL \
  --nccl-init-addr ${HEAD_NODE_IPADDRESS}:25000 \
  --nnodes 4 \
  --node-rank ${NODE_RANK} \
  --tp-size 32 \
  --context-length 131072 \
  --max-running-requests 4 \
  --cuda-graph-max-bs 4
```

> Note: This configuration reflects the current state of upstream SGLang support.
> Parameters and recommended settings may evolve as SGLang integration and benchmarking mature.

## Limitations

- A.X K1 may generate incorrect or misleading information due to its stochastic nature.
- Reasoning outputs in Think mode should not be interpreted as faithful representations of the model’s internal decision process.
- Performance may vary across domains and languages depending on data coverage.

## Citation

If you use A.X K1 in your research, please cite the technical report:

```bibtex
@techreport{axk1,
  title       = {A.X K1 Technical Report},
  author      = {{SK Telecom}},
  institution = {SK Telecom},
  year        = {2025},
  month       = {January},
  note        = {Technical report}
  url         = {https://github.com/SKT-AI/A.X-K1/releases/download/v1.0/A_X_Tech_Report.pdf}
}
```
