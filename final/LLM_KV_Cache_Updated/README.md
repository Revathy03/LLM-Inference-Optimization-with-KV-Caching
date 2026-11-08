# LLM Inference Optimization with KV Caching — corrected version

Corrected from [Revathy03/LLM-Inference-Optimization-with-KV-Caching](https://github.com/Revathy03/LLM-Inference-Optimization-with-KV-Caching), source commit `df88259`.

This folder is a complete replacement project. Extract it into a **new folder** and run commands from that folder. Keep all files together. The original repository has not been pushed to or changed remotely.

## Quick start

Use Python 3.10 or later. With your Python environment active:

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_kv_cache
python sample.py --model gpt2 --mode prefix --max-new-tokens 50
```

The first GPT-2 run downloads pretrained weights and tokenizer data; internet access and enough disk/RAM are required. An existing CUDA-enabled PyTorch installation can be retained if it satisfies requirements. The script selects CUDA when available, otherwise CPU. Default precision is float32 and default decoding is greedy for easier comparison.

Example with your own prompts (run as one line):

```bash
python sample.py --mode prefix --prompt "Explain transformers in simple terms using an analogy" --prompt "Explain transformers in simple terms focusing on attention" --max-new-tokens 30
```

Compare modes:

```bash
python sample.py --mode none --prompt "The capital of France" --max-new-tokens 30
python sample.py --mode kv --prompt "The capital of France" --max-new-tokens 30
python sample.py --mode prefix --prompt "The capital of France" --prompt "The capital of France" --max-new-tokens 30
```

Use `--temperature 0.8 --top-k 200` for sampling. `--temperature 0` means greedy decoding. Use `--attention sdpa` to select PyTorch scaled-dot-product attention; `manual` is the default for the original educational implementation. SDPA dispatches to the backend available on your device; it does not guarantee a FlashAttention kernel. CUDA examples can add `--device cuda --dtype float16`.

`--checkpoint path/to/ckpt.pt` loads a local checkpoint containing `model_args` and `model` (as in the original project). The text CLI expects GPT-2 token IDs and a GPT-2-compatible vocabulary; a character-level custom model needs its own tokenizer and should use the Python model API directly. Do not assume a custom tokenizer is interchangeable with GPT-2.

## Which files to study

| File | Purpose |
|---|---|
| `model.py` | GPT-2 architecture, attention, positional offsets, decode KV cache, cached/uncached generation |
| `radix.py` | Compressed radix tree, partial matches, correct node splitting, KV segment storage |
| `inference.py` | Prefix-cache invalidation, prompt prefill, prefix-cached generation |
| `sample.py` | Standalone command-line text generation |
| `benchmark.py` | Comparable measurements for no cache, decode cache, and prefix reuse |
| `test_kv_cache.py` | Offline correctness and regression tests |
| `CHANGELOG.md` | Original defects and their fixes |
| `VALIDATION.md`, `test_results.txt` | Verification performed for this delivery |

The old `kv-partb.py`, `kv-partc.py`, `part c/sample.py`, and `PArt c_latest/sample.py` filenames remain runnable entry points. They use the new CLI arguments above, rather than the original missing `configurator.py` interface. The two legacy `model.py` files mirror the corrected root model. Root modules are canonical; if you edit the model later, mirror that edit into the legacy copies or use only the root entry points.

## Python API

```python
import torch
from model import GPT
from inference import PrefixKVCache, generate_with_prefix_cache

model = GPT.from_pretrained("gpt2").eval()
prompt = torch.tensor([[15496, 995]], dtype=torch.long)  # GPT-2 token IDs

# Each forward call always returns three values.
logits, loss, kv = model(prompt, use_cache=True)
logits_without_cache, _, _ = model(prompt, use_cache=False)

# Generation returns prompt + completion by default.
tokens = model.generate(prompt, 20, temperature=0)

# Keep the same PrefixKVCache instance across requests to reuse prefixes.
prefix_cache = PrefixKVCache(max_cached_tokens=4096)
tokens, stats = generate_with_prefix_cache(
    model, prompt, 20, temperature=0, kv_cache=prefix_cache, return_stats=True
)
print(stats)
```

`forward()` accepts `past_kvs` as an alias for `kv`, but do not provide both. `return_all_logits=True` returns vocabulary scores for every supplied token, which is useful for correctness checks. With `use_cache=False`, the third returned value is `None` and all transformer blocks still execute.

For advanced continuation, `model.generate(..., return_cache=True)` returns `(tokens, last_sampled_token, kv)`. The cache excludes the last sampled token because it has not yet entered the model. Feed that last token with the returned cache to continue; do not feed the entire previous prompt again. The return convention is now consistent across model copies; old scripts expecting an unconditional three-value result from `generate` must opt in with `return_cache=True`.

## Correctness and scope

- Decode caching supports equally sized, unpadded batches. Prefix reuse supports **batch size 1**, one request at a time. Padding masks and concurrent serving are outside this reference implementation.
- Every layer owns its own `(K, V)` pair, shaped `[batch, heads, cached_tokens, head_dim]`.
- Cached inputs are never mutated. Position indices continue from the cached length.
- Attention applies a correctly offset causal mask with either backend, including a multi-token suffix after a cached prefix.
- Exact prompt repeats are supported. For a prompt of length `T`, at most `T-1` tokens are reused: the final prompt token is processed again to recover next-token logits. KV tensors alone do not provide those logits.
- Prefix nodes store only their token segment's tensors, not full cumulative snapshots. Tensor storage scales with unique retained prefix tokens. These are detached copies so slices do not retain large unrelated backing tensors.
- Prefix storage has a 4096-token default budget across tree branches. When an insertion would exceed it, the tree is cleared; oversized individual prompts are not retained. This is a simple clear-on-capacity policy, **not LRU eviction**. Working tensors and active decode KV are additional memory.
- A change of model, tracked parameter version, device, dtype, autocast mode, context size, or attention backend clears the prefix tree. Call `cache.clear()` after any manual untracked `.data` mutation or other semantic model change. Do not reuse raw KV across different weights or precision settings.
- Call `model.eval()` for inference. Prefix reuse and generation run under `torch.no_grad()`.
- GPT-2's supported context remains its configured limit (1024 for pretrained GPT-2). Prompt + prior cache + requested output must fit. The code raises a clear error instead of silently shifting learned absolute positions or allowing a stale cache to grow past the limit.
- No automatic EOS stopping is used: generation produces the requested number of tokens, making benchmark workloads comparable.

## Benchmark on your machine

```bash
python benchmark.py --model gpt2 --device cuda --dtype float16 --max-new-tokens 100 --repeats 5 --output benchmark_results.json
python benchmark.py --model gpt2 --device cuda --dtype float16 --max-new-tokens 100 --repeats 5 --warm-prefix-cache --output benchmark_warm.json
```

For CPU, use `--device cpu --dtype float32`. Repeated `--prompt` arguments customize the workload.

The benchmark uses the same model, prompts, output count, precision, attention backend, and greedy policy for all modes. It warms the execution paths, synchronizes CUDA around timed calls, and reports per-request generation latency, output tokens/second, cached-token counts, output parity, and PyTorch CUDA allocated/peak memory. CPU GPU-memory fields are null.

Without `--warm-prefix-cache`, each repeat starts with an empty tree and then reuses prefixes as the ordered workload progresses. With the flag, all prompts are prefetched outside the timed region. The result records this distinction. Output parity is reported rather than assumed; small floating-point differences can affect greedy selection near ties.

**Timing is generation-call latency**, including prefix lookup/insertion when applicable, but excluding model loading, tokenization, warmup, and optional cache warming. It is not a web-service end-to-end metric or a separately measured TTFT/TBT result. Peak allocated memory is PyTorch tensor allocation, not total GPU process memory. Run on an otherwise quiet GPU and use repeated measurements before making performance claims.

The original resume's **48.28% improvement has not been reproduced or assumed**. Correctness does not guarantee a speedup for every workload; Python tree traversal, cache copies, and short prompts can reduce or outweigh the gain.
