"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        if config.attention_backend not in {"manual", "sdpa"}:
            raise ValueError("attention_backend must be manual or sdpa")
        self.flash = config.attention_backend == "sdpa"
        self.register_buffer(
            "bias", torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x, kv=None, use_cache=True):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        past_len = 0
        if kv is not None:
            past_k, past_v = kv
            if past_k.dtype != k.dtype or past_v.dtype != v.dtype:
                raise ValueError("KV dtype changed; clear the cache when changing precision")
            past_len = past_k.size(2)
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        # Offset causal mask: also correct for multiple new tokens after a prefix.
        # Bare is_causal=True would align a non-square SDPA mask incorrectly here.
        mask = self.bias[:, :, past_len:past_len + T, :past_len + T]
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = self.attn_dropout(F.softmax(scores, dim=-1))
            y = weights @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, (k, v) if use_cache else None


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU(approximate="tanh") # GPT-2 gelu_new
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, kv=None, use_cache=True):
        y, new_kv = self.attn(self.ln_1(x), kv, use_cache=use_cache)
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    attention_backend: str = "manual" # "manual" or PyTorch "sdpa"

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        # Use get_num_params() when reporting model size.

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _validate_kv(self, kv, batch_size, device):
        if kv is None:
            return 0
        if len(kv) != self.config.n_layer:
            raise ValueError("KV cache must have one (K, V) pair per transformer layer")
        if all(pair is None for pair in kv):
            return 0
        if any(pair is None for pair in kv):
            raise ValueError("KV cache cannot mix empty and populated layers")
        past_len = None
        for pair in kv:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("Each layer cache must be a (K, V) pair")
            k, v = pair
            expected = (batch_size, self.config.n_head)
            if (k.ndim != 4 or v.shape != k.shape or k.shape[:2] != expected
                    or k.shape[3] != self.config.n_embd // self.config.n_head):
                raise ValueError("Invalid KV tensor shape")
            if k.device != device or v.device != device or k.dtype != v.dtype:
                raise ValueError("KV cache device/dtype does not match the input or its pair")
            if past_len is not None and k.size(2) != past_len:
                raise ValueError("All layer caches must have the same sequence length")
            past_len = k.size(2)
        return past_len

    def forward(self, idx, kv=None, targets=None, *, use_cache=True,
                past_kvs=None, position_ids=None, return_all_logits=False):
        """Return (logits, loss, new_kv); new_kv is None with use_cache=False.

        Cache inputs are read-only: this call never replaces entries in the caller's
        list or mutates the cached tensors. past_kvs is an alias for kv.
        """
        if past_kvs is not None:
            if kv is not None:
                raise ValueError("Pass either kv or past_kvs, not both")
            kv = past_kvs
        if idx.ndim != 2 or idx.size(0) == 0 or idx.size(1) == 0:
            raise ValueError("idx must have shape [batch, nonempty sequence]")
        if idx.dtype != torch.long:
            raise ValueError("idx must contain torch.long token IDs")
        if not use_cache and kv is not None:
            raise ValueError("A supplied KV cache requires use_cache=True")
        b, t = idx.shape
        past_len = self._validate_kv(kv, b, idx.device)
        if past_len + t > self.config.block_size:
            raise ValueError("Cached tokens + new tokens exceed the model context limit")
        if self.training and past_len:
            raise ValueError("Reusing past KV states requires model.eval()")
        expected_pos = torch.arange(past_len, past_len + t, device=idx.device)
        if position_ids is not None:
            if (position_ids.shape != expected_pos.shape or
                    not torch.equal(position_ids.to(idx.device), expected_pos)):
                raise ValueError("position_ids must continue exactly after the cached prefix")
        x = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(expected_pos)
        )
        previous = [None] * self.config.n_layer if kv is None else kv
        new_kv = [] if use_cache else None
        for block, layer_kv in zip(self.transformer.h, previous):
            x, pair = block(x, layer_kv, use_cache=use_cache)
            if use_cache:
                new_kv.append(pair)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x if targets is not None or return_all_logits else x[:, [-1], :])
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-1)
        return logits, loss, new_kv

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        return model.load_hf_weights(model_hf)

    @torch.no_grad()
    def load_hf_weights(self, model_hf):
        """Copy a compatible Hugging Face GPT-2 model, including Conv1D transposes."""
        sd = self.state_dict()
        sd_hf = model_hf.state_dict()
        ignored = (".attn.bias", ".attn.masked_bias")
        keys = {k for k in sd if not k.endswith(ignored)}
        hf_keys = {k for k in sd_hf if not k.endswith(ignored)}
        if keys != hf_keys:
            raise ValueError("Hugging Face and local GPT-2 weight names differ")
        transposed = ('attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight')
        for key in keys:
            weight = sd_hf[key].t() if key.endswith(transposed) else sd_hf[key]
            if sd[key].shape != weight.shape:
                raise ValueError(f"Shape mismatch for {key}: {sd[key].shape} vs {weight.shape}")
            sd[key].copy_(weight)
        return self

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, kv=None,
                 *, use_cache=True, return_cache=False):
        """Generate a fixed token count. temperature=0 selects greedy decoding.

        No silent context truncation: GPT-2's learned absolute positions make
        retaining KV after shifting a window incorrect. A clear error is raised.
        Returned KV (if requested) excludes the final sampled token, which has
        not been fed through the model yet.
        """
        validate_sampling(max_new_tokens, temperature, top_k)
        if self.training:
            raise ValueError("Call model.eval() before generating")
        if idx.ndim != 2 or idx.size(0) == 0 or idx.size(1) == 0:
            raise ValueError("Generation requires a nonempty [batch, tokens] prompt")
        if kv is not None and not use_cache:
            raise ValueError("An initial KV cache requires use_cache=True")
        past_len = self._validate_kv(kv, idx.size(0), idx.device)
        if past_len + idx.size(1) + max_new_tokens > self.config.block_size:
            raise ValueError("Prompt + cached prefix + requested output exceed context limit")
        result = idx.clone()
        current = idx
        for _ in range(max_new_tokens):
            logits, _, new_kv = self(current, kv=kv, use_cache=use_cache)
            next_token = sample_next(logits[:, -1, :], temperature, top_k)
            result = torch.cat((result, next_token), dim=1)
            kv = new_kv
            current = next_token if use_cache else result
        if return_cache:
            return result, result[:, -1:], kv
        return result


def validate_sampling(max_new_tokens, temperature, top_k):
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and >= 0 (0 means greedy)")
    if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
        raise ValueError("top_k must be a positive integer or None")


def sample_next(logits, temperature=1.0, top_k=None):
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    scores = logits / temperature
    if top_k is not None:
        values, _ = torch.topk(scores, min(top_k, scores.size(-1)))
        scores = scores.masked_fill(scores < values[:, [-1]], float("-inf"))
    return torch.multinomial(F.softmax(scores, dim=-1), num_samples=1)
