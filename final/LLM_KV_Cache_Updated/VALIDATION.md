# Validation performed

Date: 7 November 2026

- Python: 3.12.14
- PyTorch: 2.15.0+cpu
- Transformers: 4.57.6
- tiktoken: 0.14.0
- Device: CPU; CUDA was unavailable.
- Command: `python -m unittest -v test_kv_cache`
- Result: **19 tests passed, no failures or skips.** See `test_results.txt`.
- Six CLI help checks passed: root sample/benchmark, kv-partb, kv-partc, and both legacy sample entry points.

The tests compare manual and SDPA attention, cached/uncached greedy outputs, token/chunk logits, positional offsets, immutable cache inputs, repeated/partial/shorter prefixes, randomized radix branching, memory accounting/budget, model/precision invalidation, context limits, zero-length generation, and benchmark output parity.

The Hugging Face check creates a small randomly initialized GPT-2 model locally, imports its weights using the production loader, and compares logits against Hugging Face in full-context and cached modes (absolute tolerance 2e-6, relative tolerance 2e-5). This is an architecture/weight-copy correctness check, not a pretrained text-quality or GPU performance experiment.

Not performed: downloading/running full pretrained GPT-2 weights, GPU timing or memory measurements, or reproducing the resume's 48.28% improvement. Run the included benchmark on your target hardware for those measurements. Dependencies were tested at the versions above; the entire allowed version range in requirements.txt was not exhaustively tested.
