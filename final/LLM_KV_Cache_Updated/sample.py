"""Standalone GPT-2 generation; no configurator.py or notebook cells required."""
import argparse
import json
from pathlib import Path
import torch
from model import GPT, GPTConfig
from inference import PrefixKVCache, generate_with_prefix_cache


DEFAULT_PROMPTS = [
    "Explain transformers in simple terms using an analogy",
    "Explain transformers in simple terms focusing on attention",
    "Explain transformers in simple terms for beginners in AI",
    "What is life?",
    "What is life? What is the universe? How life exists on Earth?",
    "What is life?",  # An exact repeat deliberately exercises the full-hit case.
]


def add_model_args(parser):
    parser.add_argument("--model", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], default="gpt2")
    parser.add_argument("--checkpoint", type=Path, help="Optional local GPT checkpoint with model_args and model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--attention", choices=["manual", "sdpa"], default="manual")
    parser.add_argument("--seed", type=int, default=1337)


def load_model(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "cpu" and args.dtype == "float16":
        raise ValueError("Use float32 (recommended) or bfloat16 on CPU")
    torch.manual_seed(args.seed)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = GPTConfig(**checkpoint["model_args"])
        model = GPT(config)
        state = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()}
        model.load_state_dict(state)
    else:
        model = GPT.from_pretrained(args.model, {"dropout": 0.0})
    for block in model.transformer.h:
        block.attn.flash = args.attention == "sdpa"
    model.config.attention_backend = args.attention
    return model.to(device=device, dtype=getattr(torch, args.dtype)).eval()


def encode_prompts(prompts, device):
    import tiktoken
    encoder = tiktoken.get_encoding("gpt2")
    encoded = []
    for prompt in prompts:
        tokens = encoder.encode(prompt, allowed_special={"<|endoftext|>"})
        if not tokens:
            raise ValueError("Empty prompts are unsupported; use <|endoftext|> to start unconditionally")
        encoded.append(torch.tensor([tokens], device=device, dtype=torch.long))
    return encoder, encoded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--mode", choices=["none", "kv", "prefix"], default="prefix")
    parser.add_argument("--prompt", action="append", help="Repeat this option to supply multiple prompts")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = deterministic greedy decoding")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--max-cached-tokens", type=int, default=4096)
    args = parser.parse_args(argv)
    model = load_model(args)
    encoder, inputs = encode_prompts(args.prompt or DEFAULT_PROMPTS, args.device)
    cache = PrefixKVCache(args.max_cached_tokens)
    for idx in inputs:
        if args.mode == "prefix":
            output, stats = generate_with_prefix_cache(
                model, idx, args.max_new_tokens, args.temperature, args.top_k,
                cache, return_stats=True,
            )
            print(json.dumps(stats))
        else:
            output = model.generate(idx, args.max_new_tokens, args.temperature,
                                    args.top_k, use_cache=args.mode == "kv")
        print(encoder.decode(output[0].tolist()))
        print("-" * 40)
    if args.mode == "prefix":
        print("Retained prefix cache:", cache.get_stats())


if __name__ == "__main__":
    main()
