"""Distributed inference script for A.X K1."""

import argparse
import os

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed inference for EP models")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model directory")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Maximum new tokens to generate")
    parser.add_argument("--disable_kv_cache", action="store_true", help="Disable KV cache during generation")
    return parser.parse_args()


def broadcast_inputs(inputs: dict | None, rank: int) -> dict:
    """Broadcast tokenized inputs from rank 0 to all ranks."""
    # Filter to only required keys on rank 0
    if rank == 0:
        inputs = {k: inputs[k] for k in ("input_ids", "attention_mask") if k in inputs}
        shape = torch.tensor(inputs["input_ids"].shape, device="cuda")
    else:
        shape = torch.zeros(2, dtype=torch.long, device="cuda")

    dist.broadcast(shape, src=0)
    seq_shape = tuple(shape.tolist())

    if rank != 0:
        inputs = {
            "input_ids": torch.zeros(seq_shape, dtype=torch.long, device="cuda"),
            "attention_mask": torch.ones(seq_shape, dtype=torch.long, device="cuda"),
        }

    for tensor in inputs.values():
        dist.broadcast(tensor, src=0)

    return inputs


@torch.inference_mode()
def main():
    args = parse_args()

    # Initialize distributed environment
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank, world_size = dist.get_rank(), dist.get_world_size()

    # Load model with Expert Parallelism (EP = world_size)
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    config.ep_size = world_size
    config.use_cache = not args.disable_kv_cache

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        config=config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).cuda().eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # Prepare inputs on rank 0 and broadcast to all ranks
    if rank == 0:
        messages = [{"role": "user", "content": args.prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
            enable_thinking=True,
            return_token_type_ids=False,
        ).to("cuda")
    else:
        inputs = None

    inputs = broadcast_inputs(inputs, rank)

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        use_cache=config.use_cache,
    )

    if rank == 0:
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    dist.barrier()


if __name__ == "__main__":
    main()
