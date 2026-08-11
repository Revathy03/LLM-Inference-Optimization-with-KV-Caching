# """
# Sample from a trained model
# """
# import os
# import pickle
# from contextlib import nullcontext
# import torch
# import tiktoken
# from model import GPTConfig, GPT

# import time
# from torch.nn import functional as F

# # -----------------------------------------------------------------------------
# init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
# out_dir = 'out' # ignored if init_from is not 'resume'
# start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
# num_samples = 10 # number of samples to draw
# max_new_tokens = 500 # number of tokens generated in each sample
# temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
# top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
# seed = 1337
# device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
# compile = False # use PyTorch 2.0 to compile the model to be faster
# exec(open('configurator.py').read()) # overrides from command line or config file
# # -----------------------------------------------------------------------------

# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
# torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
# device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# # model
# if init_from == 'resume':
#     # init from a model saved in a specific directory
#     ckpt_path = os.path.join(out_dir, 'ckpt.pt')
#     checkpoint = torch.load(ckpt_path, map_location=device)
#     gptconf = GPTConfig(**checkpoint['model_args'])
#     model = GPT(gptconf)
#     state_dict = checkpoint['model']
#     unwanted_prefix = '_orig_mod.'
#     for k,v in list(state_dict.items()):
#         if k.startswith(unwanted_prefix):
#             state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
#     model.load_state_dict(state_dict)
# elif init_from.startswith('gpt2'):
#     # init from a given GPT-2 model
#     model = GPT.from_pretrained(init_from, dict(dropout=0.0))

# model.eval()
# model.to(device)
# if compile:
#     model = torch.compile(model) # requires PyTorch 2.0 (optional)

# # look for the meta pickle in case it is available in the dataset folder
# load_meta = False
# if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
#     meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
#     load_meta = os.path.exists(meta_path)
# if load_meta:
#     print(f"Loading meta from {meta_path}...")
#     with open(meta_path, 'rb') as f:
#         meta = pickle.load(f)
#     # TODO want to make this more general to arbitrary encoder/decoder schemes
#     stoi, itos = meta['stoi'], meta['itos']
#     encode = lambda s: [stoi[c] for c in s]
#     decode = lambda l: ''.join([itos[i] for i in l])
# else:
#     # ok let's assume gpt-2 encodings by default
#     print("No meta.pkl found, assuming GPT-2 encodings...")
#     enc = tiktoken.get_encoding("gpt2")
#     encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
#     decode = lambda l: enc.decode(l)


# # encode the beginning of the prompt
# if start.startswith('FILE:'):
#     with open(start[5:], 'r', encoding='utf-8') as f:
#         start = f.read()




# class PrefixKVNode:
#     def __init__(self):
#         self.children = {}        # token_id -> PrefixKVNode
#         self.past_kvs = None      # KV cache for THIS prefix



# class PrefixKVCache:
#     def __init__(self):
#         self.root = PrefixKVNode()

#     def find_longest_prefix(self, tokens):
#         """
#         tokens: List[int]
#         returns:
#           matched_len, node
#         """
#         node = self.root
#         matched = 0
#         for tok in tokens:
#             if tok in node.children:
#                 node = node.children[tok]
#                 matched += 1
#             else:
#                 break
#         return matched, node



# @torch.no_grad()
# def generate_with_prefix_cache(
#     model,
#     idx,
#     max_new_tokens,
#     temperature=1.0,
#     top_k=None,
#     kv_cache=None
# ):
#     device = idx.device
#     tokens = idx[0].tolist()  # batch=1 assumed
#     print(tokens)

#     # 1️⃣ Find longest prefix already cached
#     matched_len, node = kv_cache.find_longest_prefix(tokens)

#     if node.past_kvs is not None:
#         past_kvs = node.past_kvs
#         print(f"Reusing KV cache for prefix length {matched_len}")
#     else:
#         past_kvs = None
#         print("No KV cache found")

#     # 2️⃣ Compute KV for remaining prefix tokens
#     cur_len = matched_len

#     node_ptr = node  # start from last matched node

#     for i in range(matched_len, len(tokens)):
#         tok = tokens[i]
    
#         input_token = torch.tensor([[tok]], dtype=torch.long, device=device)
#         pos_ids = torch.tensor([cur_len], dtype=torch.long, device=device)
    
#         _, _, past_kvs = model(
#             input_token,
#             past_kvs=past_kvs,
#             position_ids=pos_ids,
#             use_cache=True
#         )
    
#         # 🔹 INSERT PREFIX KV HERE
#         if tok not in node_ptr.children:
#             node_ptr.children[tok] = PrefixKVNode()
    
#         node_ptr = node_ptr.children[tok]
    
#         # store KV snapshot for this prefix
#         node_ptr.past_kvs = past_kvs
    
#         cur_len += 1


#     # 3️⃣ Autoregressive generation
#     for _ in range(max_new_tokens):
#         input_token = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
#         pos_ids = torch.tensor([cur_len], dtype=torch.long, device=device)

#         logits, _, past_kvs = model(
#             input_token,
#             past_kvs=past_kvs,
#             position_ids=pos_ids,
#             use_cache=True
#         )

#         logits = logits[:, -1, :] / temperature

#         if top_k is not None:
#             v, _ = torch.topk(logits, top_k)
#             logits[logits < v[:, [-1]]] = -float("inf")

#         probs = F.softmax(logits, dim=-1)
#         next_token = torch.multinomial(probs, num_samples=1)

#         tokens.append(next_token.item())
#         cur_len += 1

#     return torch.tensor(tokens, device=device).unsqueeze(0)





# # --------------------------------------------------
# # Prompts (NO prompt is full prefix of another)
# # --------------------------------------------------
# prompts = [
#     "Explain transformers in simple terms using an analogy",
#     "Explain transformers in simple terms focusing on attention",
#     "Explain transformers in simple terms for beginners in AI",
#     "What is life?",
#     "What is life? What is the universe? How life exists on Earth?"
# ]

# # --------------------------------------------------
# # Run generation (KV reuse happens automatically)
# # --------------------------------------------------
# kv_cache = PrefixKVCache()

# with torch.no_grad():
#     with ctx:
#         for i, prompt in enumerate(prompts):
#             print(f"\n================ Prompt {i+1} ================")
#             print(prompt)

#             tokens = encode(prompt)
#             x = torch.tensor(tokens, device=device).unsqueeze(0)

#             y = generate_with_prefix_cache(
#                 model,
#                 x,
#                 max_new_tokens=100,
#                 temperature=temperature,
#                 top_k=top_k,
#                 kv_cache=kv_cache
#             )

#             print("\n--- Output ---")
#             print(decode(y[0].tolist()))

%%writefile sample.py
"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

import time
from torch.nn import functional as F

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)


# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()




class PrefixKVNode:
    def __init__(self):
        self.children = {}        # token_id -> PrefixKVNode
        self.past_kvs = None      # KV cache for THIS prefix



class PrefixKVCache:
    def __init__(self):
        self.root = PrefixKVNode()

    def find_longest_prefix(self, tokens):
        """
        tokens: List[int]
        returns:
          matched_len, node
        """
        node = self.root
        matched = 0
        for tok in tokens:
            if tok in node.children:
                node = node.children[tok]
                matched += 1
            else:
                break
        return matched, node



@torch.no_grad()
def generate_with_prefix_cache(
    model,
    idx,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    kv_cache=None
):
    device = idx.device
    tokens = idx[0].tolist()  # batch=1 assumed
    print(tokens)

    # 1️⃣ Find longest prefix already cached
    matched_len, node = kv_cache.find_longest_prefix(tokens) #return back here #1

    if node.past_kvs is not None:
        past_kvs = node.past_kvs
        print(f"Reusing KV cache for prefix length {matched_len}")
    else:
        past_kvs = None
        print("No KV cache found")

    # 2️⃣ Compute KV for remaining prefix tokens
    cur_len = matched_len

    node_ptr = node  # start from last matched node

    last_logit = None

    for i in range(matched_len, len(tokens)):
        tok = tokens[i]
    
        input_token = torch.tensor([[tok]], dtype=torch.long, device=device)
        pos_ids = torch.tensor([cur_len], dtype=torch.long, device=device)
    
        logits, _, past_kvs = model(
            input_token,
            past_kvs=past_kvs,
            position_ids=pos_ids,
            use_cache=True
        )
    
        # 🔹 INSERT PREFIX KV HERE
        if tok not in node_ptr.children:
            node_ptr.children[tok] = PrefixKVNode()
    
        node_ptr = node_ptr.children[tok]
    
        # store KV snapshot for this prefix
        node_ptr.past_kvs = past_kvs

        last_logit = logits
        cur_len += 1

    logits = last_logit[:, -1, :] / temperature

    if top_k is not None:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("inf")

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    tokens.append(next_token.item())

    # 3️⃣ Autoregressive generation
    for _ in range(max_new_tokens - 1):
        input_token = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
        pos_ids = torch.tensor([cur_len], dtype=torch.long, device=device)

        logits, _, past_kvs = model(
            input_token,
            past_kvs=past_kvs,
            position_ids=pos_ids,
            use_cache=True
        )

        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        tokens.append(next_token.item())
        cur_len += 1

    return torch.tensor(tokens, device=device).unsqueeze(0)





# --------------------------------------------------
# Prompts (NO prompt is full prefix of another)
# --------------------------------------------------
prompts = [
    "Explain transformers in simple terms using an analogy",
    "Explain transformers in simple terms focusing on attention",
    "Explain transformers in simple terms for beginners in AI",
    "What is life?",
    "What is life? What is the universe? How life exists on Earth?",
    "Revathy is a ",
    "abc ka badiya"
]

# --------------------------------------------------
# Run generation (KV reuse happens automatically)
# --------------------------------------------------
kv_cache = PrefixKVCache()

with torch.no_grad():
    with ctx:
        for i, prompt in enumerate(prompts):
            print(f"\n================ Prompt {i+1} ================")
            print(prompt)

            tokens = encode(prompt)
            x = torch.tensor(tokens, device=device).unsqueeze(0) # [T] -> [1,]

            y = generate_with_prefix_cache(
                model,
                x,
                max_new_tokens=100,
                temperature=temperature,
                top_k=top_k,
                kv_cache=kv_cache
            )

            print("\n--- Output ---")
            print(decode(y[0].tolist()))



