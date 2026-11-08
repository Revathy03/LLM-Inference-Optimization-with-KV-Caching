"""Offline correctness tests: small random GPTs, no pretrained download or GPU."""
import ast
import contextlib
import importlib.util
import io
import random
import unittest
from pathlib import Path
import torch
from model import GPT, GPTConfig
from inference import PrefixKVCache, generate_with_prefix_cache
from radix import RadixTree
from benchmark import benchmark

torch.set_num_threads(1)


def tiny_model(backend="manual"):
    torch.manual_seed(123)
    return GPT(GPTConfig(block_size=32, vocab_size=41, n_layer=2,
                         n_head=2, n_embd=16, dropout=0.0,
                         attention_backend=backend)).eval()


def ids(tokens):
    return torch.tensor([tokens], dtype=torch.long)


class CacheCorrectnessTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.prompt = ids([1, 2, 3, 4, 5, 6])

    def assertCacheEqual(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)

    @torch.no_grad()
    def test_token_and_chunk_decoding_match_full_context(self):
        for backend in ("manual", "sdpa"):
            model = tiny_model(backend)
            prompt = torch.cat((self.prompt, self.prompt.flip(1)), dim=0)
            for split in (1, 3, 5):
                with self.subTest(backend=backend, split=split):
                    full, _, full_cache = model(prompt, return_all_logits=True)
                    _, _, past = model(prompt[:, :split])
                    logits, _, cache = model(prompt[:, split:], kv=past, return_all_logits=True)
                    torch.testing.assert_close(logits, full[:, split:], atol=2e-6, rtol=2e-5)
                    self.assertCacheEqual(cache, full_cache)

    @torch.no_grad()
    def test_cached_and_uncached_greedy_outputs_match(self):
        for backend in ("manual", "sdpa"):
            model = tiny_model(backend)
            for prompt in (self.prompt, torch.cat((self.prompt, self.prompt.flip(1)), dim=0)):
                uncached = model.generate(prompt, 8, temperature=0, use_cache=False)
                cached = model.generate(prompt, 8, temperature=0)
                self.assertTrue(torch.equal(uncached, cached))

    @torch.no_grad()
    def test_input_cache_is_not_mutated(self):
        _, _, old = self.model(self.prompt[:, :3])
        saved = [(k.clone(), v.clone()) for k, v in old]
        old_ids = [(id(k), id(v)) for k, v in old]
        self.model(self.prompt[:, 3:], kv=old)
        self.assertEqual(old_ids, [(id(k), id(v)) for k, v in old])
        self.assertCacheEqual(old, saved)

    @torch.no_grad()
    def test_position_ids_are_offset_and_validated(self):
        _, _, past = self.model(self.prompt[:, :3])
        automatic, _, _ = self.model(self.prompt[:, 3:], kv=past)
        explicit, _, _ = self.model(self.prompt[:, 3:], past_kvs=past, position_ids=torch.arange(3, 6))
        torch.testing.assert_close(automatic, explicit)
        with self.assertRaises(ValueError):
            self.model(self.prompt[:, 3:], kv=past, position_ids=torch.arange(3))

    def test_uncached_training_visits_transformer_blocks(self):
        self.model.train()
        _, loss, cache = self.model(self.prompt, targets=self.prompt, use_cache=False)
        self.assertIsNone(cache)
        loss.backward()
        grad = self.model.transformer.h[0].attn.c_attn.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_zero_tokens_and_invalid_sampling(self):
        self.assertTrue(torch.equal(self.prompt, self.model.generate(self.prompt, 0)))
        output = generate_with_prefix_cache(self.model, self.prompt, 0)
        self.assertTrue(torch.equal(self.prompt, output))
        for kwargs in ({"max_new_tokens": -1}, {"max_new_tokens": 1, "temperature": -1},
                       {"max_new_tokens": 1, "temperature": float("nan")},
                       {"max_new_tokens": 1, "top_k": 0}):
            with self.assertRaises(ValueError):
                self.model.generate(self.prompt, **kwargs)
        self.model.generate(self.prompt, 2, top_k=1000)

    @torch.no_grad()
    def test_context_boundaries(self):
        prompt = ids([1] * 30)
        self.assertEqual(self.model.generate(prompt, 2, temperature=0).size(1), 32)
        with self.assertRaises(ValueError):
            self.model.generate(prompt, 3)
        with self.assertRaises(ValueError):
            generate_with_prefix_cache(self.model, prompt, 3)
        _, _, past = self.model(ids([1] * 32))
        with self.assertRaises(ValueError):
            self.model(ids([2]), kv=past)

    @torch.no_grad()
    def test_bad_cache_and_empty_inputs(self):
        _, _, past = self.model(self.prompt)
        for bad in (past[:1], [None, past[1]], [past[0], (past[1][0][:, :, :-1], past[1][1][:, :, :-1])]):
            with self.assertRaises(ValueError):
                self.model(ids([1]), kv=bad)
        with self.assertRaises(ValueError):
            self.model(ids([]))
        with self.assertRaises(ValueError):
            self.model.generate(ids([]), 2)
        with self.assertRaises(ValueError):
            self.model(ids([1]), kv=past, use_cache=False)

    @torch.no_grad()
    def test_generation_can_continue_from_returned_cache(self):
        first, last, kv = self.model.generate(self.prompt, 2, temperature=0, return_cache=True)
        continuation = self.model.generate(last, 3, temperature=0, kv=kv)
        combined = torch.cat((first[:, :-1], continuation), dim=1)
        full = self.model.generate(self.prompt, 5, temperature=0)
        self.assertTrue(torch.equal(combined, full))

    @torch.no_grad()
    def test_prefix_reuse_exact_partial_shorter_and_extensions(self):
        prompts = [[1, 2, 3, 4, 5], [1, 2, 7, 8], [1, 2, 3], [1, 2, 3, 4, 5],
                   [1, 2, 3, 4, 5, 9], [1], [9, 8, 7], [1, 2, 7, 8]]
        for backend in ("manual", "sdpa"):
            model, cache = tiny_model(backend), PrefixKVCache()
            for tokens in prompts:
                prompt = ids(tokens)
                expected = model.generate(prompt, 4, temperature=0, use_cache=False)
                actual = generate_with_prefix_cache(model, prompt, 4, temperature=0, kv_cache=cache)
                self.assertTrue(torch.equal(expected, actual), (backend, tokens))
                full, _, full_kv = model(prompt)
                prefix, prefix_kv, matched = cache.prefill(model, prompt)
                self.assertEqual(matched, len(tokens) - 1)
                torch.testing.assert_close(prefix, full, atol=2e-6, rtol=2e-5)
                self.assertCacheEqual(prefix_kv, full_kv)

    @torch.no_grad()
    def test_prefix_cache_does_not_grow_on_exact_repeat(self):
        cache = PrefixKVCache()
        cache.prefill(self.model, self.prompt)
        before = cache.get_stats()
        generate_with_prefix_cache(self.model, self.prompt, 5, kv_cache=cache)
        self.assertEqual(before, cache.get_stats())
        per_token = 2 * 2 * 16 * 4  # K/V * layers * embedding size * float32 bytes
        self.assertEqual(before["kv_bytes"], self.prompt.size(1) * per_token)

    @torch.no_grad()
    def test_cache_invalidation_after_weight_or_model_change(self):
        cache = PrefixKVCache()
        cache.prefill(self.model, self.prompt)
        self.assertEqual(cache.prefill(self.model, self.prompt)[2], 5)
        self.model.transformer.wte.weight.add_(0.01)
        self.assertEqual(cache.prefill(self.model, self.prompt)[2], 0)
        other = tiny_model()
        self.assertEqual(cache.prefill(other, self.prompt)[2], 0)
        other.double()
        self.assertEqual(cache.prefill(other, self.prompt)[2], 0)

    @torch.no_grad()
    def test_prefix_cache_budget(self):
        cache = PrefixKVCache(max_cached_tokens=6)
        for tokens in ([1, 2, 3, 4], [1, 2, 8, 9], [7, 6, 5, 4], [1] * 7):
            prompt = ids(tokens)
            actual = generate_with_prefix_cache(self.model, prompt, 2, temperature=0, kv_cache=cache)
            expected = self.model.generate(prompt, 2, temperature=0)
            self.assertTrue(torch.equal(actual, expected))
            self.assertLessEqual(cache.get_stats()["cached_tokens"], 6)

    @torch.no_grad()
    def test_radix_splits_and_randomized_longest_prefixes(self):
        tree, inserted = RadixTree(max_cached_tokens=None), []
        rng = random.Random(5)
        sequences = [[1, 2, 3, 4], [1, 2], [1, 2, 9], [1], [1, 2, 3, 8]]
        sequences += [[rng.randrange(1, 6) for _ in range(rng.randrange(1, 9))] for _ in range(40)]
        for seq in sequences:
            _, _, full_kv = self.model(ids(seq))
            tree.insert(seq, full_kv)
            inserted.append(seq)
            for query in (seq, seq[:-1], [1, 2, 3, 8, 9], [9]):
                expected_len = max((RadixTree._common(query, s) for s in inserted), default=0)
                matched, kv = tree.match(query)
                self.assertEqual(matched, expected_len)
                if matched:
                    _, _, expected_kv = self.model(ids(query[:matched]))
                    self.assertCacheEqual(kv, expected_kv)
                    kv[0][0].zero_()  # Lookup must not expose mutable tree storage.
                    self.assertCacheEqual(tree.match(query)[1], expected_kv)
            distinct_prefixes = {tuple(s[:i]) for s in inserted for i in range(1, len(s) + 1)}
            self.assertEqual(tree.get_stats()["cached_tokens"], len(distinct_prefixes))

    def test_prefix_requires_eval_and_batch_one(self):
        with self.assertRaises(ValueError):
            PrefixKVCache().prefill(self.model.train(), self.prompt)
        self.model.eval()
        with self.assertRaises(ValueError):
            generate_with_prefix_cache(self.model, self.prompt.repeat(2, 1), 2)

    def test_benchmark_reports_all_modes_and_output_parity(self):
        for warm in (False, True):
            report = benchmark(self.model, [ids([1, 2, 3]), ids([1, 2, 4]), ids([1, 2, 3])],
                               new_tokens=2, repeats=1, warm_prefix=warm)
            self.assertEqual(len(report["requests"]), 9)
            self.assertTrue(all(r["matches_no_cache_output"] for r in report["requests"]))
            self.assertEqual(set(report["summary"]), {"none", "kv", "prefix"})
            self.assertTrue(all(r["generation_seconds"] > 0 for r in report["requests"]))

    def test_every_python_file_parses(self):
        for path in Path(__file__).parent.rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "Install transformers for HF parity")
    @torch.no_grad()
    def test_huggingface_weight_loading_and_logits(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config(vocab_size=41, n_positions=32, n_embd=16, n_layer=2,
                            n_head=2, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
                            activation_function="gelu_new")
        config._attn_implementation = "eager"
        torch.manual_seed(789)
        reference = GPT2LMHeadModel(config).eval()
        expected = reference(self.prompt, use_cache=False).logits
        for backend in ("manual", "sdpa"):
            model = tiny_model(backend).load_hf_weights(reference)
            actual, _, _ = model(self.prompt, use_cache=False, return_all_logits=True)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            _, _, kv = model(self.prompt[:, :3])
            actual, _, _ = model(self.prompt[:, 3:], kv=kv, return_all_logits=True)
            torch.testing.assert_close(actual, expected[:, 3:], atol=2e-6, rtol=2e-5)

    @torch.no_grad()
    def test_autocast_precision_change_invalidates_prefix(self):
        cache = PrefixKVCache()
        cache.prefill(self.model, self.prompt)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits, _, reused = cache.prefill(self.model, self.prompt)
            self.assertEqual(reused, 0)
            self.assertEqual(logits.dtype, torch.bfloat16)
            actual = generate_with_prefix_cache(self.model, self.prompt, 3, temperature=0, kv_cache=cache)
            expected = self.model.generate(self.prompt, 3, temperature=0, use_cache=False)
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(cache.prefill(self.model, self.prompt)[2], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
