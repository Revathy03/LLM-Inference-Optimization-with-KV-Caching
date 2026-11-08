"""Compare no cache, decoding KV cache, and radix prefix reuse on the same model.

Measures generation-call latency, excluding model loading and tokenization.
CUDA work is synchronized before/after each timed call. No speedup is assumed.
"""
import argparse
import gc
import json
import platform
import statistics
import time
from pathlib import Path
import torch
from inference import PrefixKVCache, generate_with_prefix_cache
from sample import add_model_args, load_model, encode_prompts, DEFAULT_PROMPTS


def benchmark(model, inputs, new_tokens=32, repeats=3, warm_prefix=False):
    if repeats < 1 or new_tokens < 1 or not inputs:
        raise ValueError("Use positive repeats/new_tokens and at least one prompt")
    device = next(model.parameters()).device
    cuda = device.type == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize(device)

    # Warm each path using a throwaway tree, never the measured cache.
    for mode in ("none", "kv", "prefix"):
        if mode == "prefix":
            generate_with_prefix_cache(model, inputs[0], new_tokens, temperature=0,
                                       kv_cache=PrefixKVCache())
        else:
            model.generate(inputs[0], new_tokens, temperature=0, use_cache=mode == "kv")
    sync()
    rows, references = [], {}
    for mode in ("none", "kv", "prefix"):
        for repeat in range(repeats):
            cache = PrefixKVCache()
            if mode == "prefix" and warm_prefix:
                for idx in inputs:
                    cache.prefill(model, idx)
            gc.collect()
            for prompt_id, idx in enumerate(inputs):
                sync()
                before = torch.cuda.memory_allocated(device) if cuda else None
                if cuda:
                    torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                if mode == "prefix":
                    output, stats = generate_with_prefix_cache(
                        model, idx, new_tokens, temperature=0, kv_cache=cache,
                        return_stats=True,
                    )
                else:
                    output = model.generate(idx, new_tokens, temperature=0, use_cache=mode == "kv")
                    stats = {"reused_tokens": 0}
                sync()
                elapsed = time.perf_counter() - start
                peak = torch.cuda.max_memory_allocated(device) if cuda else None
                generated = output.cpu().tolist()
                key = (repeat, prompt_id)
                if mode == "none":
                    references[key] = generated
                rows.append({
                    "mode": mode, "repeat": repeat, "prompt_id": prompt_id,
                    "prompt_tokens": idx.size(1), "output_tokens": new_tokens,
                    "reused_prefix_tokens": stats["reused_tokens"],
                    "generation_seconds": elapsed, "output_tokens_per_second": new_tokens / elapsed,
                    "gpu_allocated_before_bytes": before,
                    "gpu_peak_allocated_bytes": peak,
                    "gpu_peak_extra_bytes": peak - before if cuda else None,
                    "matches_no_cache_output": generated == references[key],
                })
                del output
    summary = {}
    baseline = statistics.mean(r["generation_seconds"] for r in rows if r["mode"] == "none")
    for mode in ("none", "kv", "prefix"):
        subset = [r for r in rows if r["mode"] == mode]
        mean = statistics.mean(r["generation_seconds"] for r in subset)
        summary[mode] = {
            "mean_generation_seconds": mean,
            "median_generation_seconds": statistics.median(r["generation_seconds"] for r in subset),
            "time_reduction_percent_vs_no_cache": 100 * (baseline - mean) / baseline,
            "all_outputs_match_no_cache": all(r["matches_no_cache_output"] for r in subset),
        }
    return {"environment": {"torch": torch.__version__, "python": platform.python_version(),
                            "device": str(device), "dtype": str(next(model.parameters()).dtype),
                            "gpu": torch.cuda.get_device_name(device) if cuda else None,
                            "attention_backend": model.config.attention_backend},
            "settings": {"repeats": repeats, "new_tokens": new_tokens,
                         "warm_prefix": warm_prefix, "temperature": 0,
                         "cache_reset_each_repeat": True,
                         "timing_excludes": ["model loading", "tokenization", "warmup", "optional prefix warming"]},
            "summary": summary, "requests": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warm-prefix-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    args = parser.parse_args(argv)
    model = load_model(args)
    _, inputs = encode_prompts(args.prompt or DEFAULT_PROMPTS, args.device)
    report = benchmark(model, inputs, args.max_new_tokens, args.repeats, args.warm_prefix_cache)
    report["settings"]["model"] = args.model
    report["settings"]["checkpoint"] = str(args.checkpoint) if args.checkpoint else None
    report["settings"]["prompts"] = args.prompt or DEFAULT_PROMPTS
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
