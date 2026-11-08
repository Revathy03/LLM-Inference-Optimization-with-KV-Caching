"""Legacy entry point backed by the corrected root modules."""
from model import GPT, GPTConfig, Block, CausalSelfAttention, LayerNorm, MLP
from inference import PrefixKVCache, generate_with_prefix_cache

if __name__ == "__main__":
    import sys
    from sample import main
    main(["--mode", "prefix", *sys.argv[1:]])
