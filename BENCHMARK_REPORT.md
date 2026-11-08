# KV-Cache Optimization Benchmark Report

## Overview
This report documents the performance improvement achieved by implementing KV-cache optimization in the nanoGPT-based transformer decoder for autoregressive token generation.

## Benchmark Configuration
- **Model**: GPT with 4 transformer layers, 4 attention heads, 64 embedding dimensions
- **Generated tokens**: 16 tokens per prompt
- **Test runs**: 5 iterations per configuration (median reported)
- **Device**: CPU
- **Implementation**: [kv-partb.py](kv-partb.py)

## Results Summary

| Prompt Length | Without KV Cache | With KV Cache | Time Reduction | Speedup |
|---:|---:|---:|---:|---:|
| 8 tokens | 0.009663 s | 0.006634 s | 31.35% | 1.46× |
| 16 tokens | 0.010215 s | 0.006732 s | 34.10% | 1.52× |
| 32 tokens | 0.011374 s | 0.006872 s | 39.58% | 1.65× |
| 64 tokens | 0.012952 s | 0.006699 s | 48.28% | 1.93× |

## Key Findings

### 1. Consistent Improvement Across All Prompt Lengths
KV caching shows measurable improvements even at short prompts (31% at 8 tokens), demonstrating the technique's fundamental effectiveness.

### 2. Scaling with Prompt Length
The improvement increases with longer prompts:
- 8→16 tokens: improvement grows from 31.35% to 34.10%
- 16→32 tokens: improvement grows from 34.10% to 39.58%
- 32→64 tokens: improvement grows from 39.58% to 48.28%

This follows the expected trend: as the prompt grows, the overhead of recomputing full sequence attention without KV cache becomes more pronounced.

### 3. Nearly 2× Speedup at 64-Token Prompt
At the longest tested prompt length, the KV-cached version is **1.93× faster** than the baseline, achieving a 48.28% reduction in total generation time.

## Technical Explanation

### Without KV Cache
At each generation step, the model reprocesses the entire sequence:
```
Step 1: forward pass on 64 tokens  → outputs token 1
Step 2: forward pass on 65 tokens  → outputs token 2
Step 3: forward pass on 66 tokens  → outputs token 3
...
Step 16: forward pass on 79 tokens → outputs token 16
```

Total tokens processed: 64 + 65 + 66 + ... + 79 = 1,152 token passes

### With KV Cache
The model processes only the new token and reuses cached attention states:
```
Step 1: forward pass on 64 tokens (cache K,V)  → outputs token 1
Step 2: forward pass on 1 token + cached K,V   → outputs token 2
Step 3: forward pass on 1 token + cached K,V   → outputs token 3
...
Step 16: forward pass on 1 token + cached K,V  → outputs token 16
```

Total tokens processed: 64 + 1 + 1 + ... + 1 = 79 token passes

**Reduction ratio**: (1,152 - 79) / 1,152 ≈ 93% fewer token passes, translating to 48% wall-clock time savings at 64 tokens.

## Conclusion

KV caching is a highly effective optimization for autoregressive LLM decoding. For a typical use case with moderately long prompts (50+ tokens), users can expect roughly **1.5–2× faster inference** with negligible impact on output quality. The technique is now standard in production LLM deployments and is essential for real-time inference applications.

## Recommendations for Further Optimization

1. **GPU acceleration**: Results on GPU would show even larger improvements, as memory bandwidth becomes the limiting factor.
2. **Shared-prefix reuse (Part C)**: For batched prompts with common prefixes, additional speedup of 2–10× is possible by reusing KV caches across prompts.
3. **Flash Attention**: Using Flash Attention (PyTorch 2.0+) would accelerate both cached and uncached paths but would amplify the relative benefit of caching.

---
**Generated**: 2026-08-16  
**Project**: PA1 – KV-Cache Optimization for Transformers
