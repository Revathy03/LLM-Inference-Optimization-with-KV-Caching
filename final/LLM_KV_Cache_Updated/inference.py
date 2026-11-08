"""Shared-prefix prefill followed by ordinary KV-cached autoregressive decoding."""
import weakref
import torch
from model import sample_next, validate_sampling
from radix import RadixTree


class PrefixKVCache:
    def __init__(self, max_cached_tokens=4096):
        self.tree = RadixTree(max_cached_tokens)
        self._owner = None
        self._signature = None

    def clear(self):
        self.tree.clear()

    def _bind(self, model):
        # Prevent accidentally reusing activations from another model, a changed
        # state_dict, device, precision, or attention backend. Use no_grad (not
        # inference_mode) when constructing this reference model's parameters.
        device_type = next(model.parameters()).device.type
        autocast = torch.is_autocast_enabled(device_type)
        signature = (
            tuple((id(p), p._version, str(p.device), p.dtype) for p in model.parameters()),
            model.config.block_size,
            tuple(block.attn.flash for block in model.transformer.h),
            autocast, torch.get_autocast_dtype(device_type) if autocast else None,
        )
        if (self._owner is None or self._owner() is not model or
                signature != self._signature):
            self.clear()
            self._owner = weakref.ref(model)
            self._signature = signature

    @torch.no_grad()
    def prefill(self, model, idx):
        if model.training:
            raise ValueError("Prefix reuse requires model.eval()")
        if idx.ndim != 2 or idx.size(0) != 1 or idx.size(1) < 1:
            raise ValueError("Prefix caching requires one nonempty prompt, shape [1, T]")
        if idx.size(1) > model.config.block_size:
            raise ValueError("Prompt exceeds the context limit")
        self._bind(model)
        tokens = idx[0].tolist()
        # Always process at least the final prompt token to obtain next-token
        # logits. This handles exact hits without saving logits for every prefix.
        matched, past_kvs = self.tree.match(tokens[:-1])
        logits, _, cache = model(idx[:, matched:], kv=past_kvs, use_cache=True)
        self.tree.insert(tokens, cache)
        return logits, cache, matched

    def get_stats(self):
        return self.tree.get_stats()


@torch.no_grad()
def generate_with_prefix_cache(model, idx, max_new_tokens, temperature=1.0,
                               top_k=None, kv_cache=None, *, return_stats=False):
    validate_sampling(max_new_tokens, temperature, top_k)
    if model.training:
        raise ValueError("Call model.eval() before generating")
    if idx.ndim != 2 or idx.size(0) != 1 or idx.size(1) < 1:
        raise ValueError("Prefix generation requires a nonempty [1, T] prompt")
    if idx.size(1) + max_new_tokens > model.config.block_size:
        raise ValueError("Prompt + requested output exceed the context limit")
    result = idx.clone()
    if max_new_tokens == 0:
        stats = {"reused_tokens": 0, "prefill_tokens": 0, "generated_tokens": 0}
        return (result, stats) if return_stats else result
    if kv_cache is None:
        kv_cache = PrefixKVCache()
    logits, past_kvs, matched = kv_cache.prefill(model, idx)
    for step in range(max_new_tokens):
        next_token = sample_next(logits[:, -1, :], temperature, top_k)
        result = torch.cat((result, next_token), dim=1)
        if step + 1 < max_new_tokens:
            logits, _, past_kvs = model(next_token, kv=past_kvs, use_cache=True)
    stats = {"reused_tokens": matched, "prefill_tokens": idx.size(1) - matched,
             "generated_tokens": max_new_tokens}
    return (result, stats) if return_stats else result
