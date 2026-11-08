"""Compressed token radix tree storing only the KV segment owned by each node.

Single-process, single-request-at-a-time reference implementation. Prefix cache
invalidation and inference integration live in inference.py.
"""
from dataclasses import dataclass, field
import torch


def slice_kv(cache, start, end, *, copy=False):
    result = []
    for k, v in cache:
        k, v = k[:, :, start:end, :], v[:, :, start:end, :]
        if copy:
            k, v = k.detach().clone(), v.detach().clone()
        result.append((k, v))
    return result


def concatenate_kv_caches(caches):
    if not caches:
        return None
    # Copy even one segment: callers must not be able to corrupt tree storage.
    return [
        (torch.cat([c[layer][0] for c in caches], dim=2),
         torch.cat([c[layer][1] for c in caches], dim=2))
        for layer in range(len(caches[0]))
    ]


@dataclass
class RadixTreeNode:
    token_sequence: list = field(default_factory=list)
    kv_cache: object = None
    children: dict = field(default_factory=dict)


class RadixTree:
    def __init__(self, max_cached_tokens=4096):
        if max_cached_tokens is not None and max_cached_tokens < 1:
            raise ValueError("max_cached_tokens must be positive or None")
        self.max_cached_tokens = max_cached_tokens
        self.clear()

    def clear(self):
        self.root = RadixTreeNode()
        self.cached_tokens = 0

    @staticmethod
    def _common(a, b):
        i = 0
        while i < min(len(a), len(b)) and a[i] == b[i]:
            i += 1
        return i

    def _walk(self, tokens):
        node, offset, segments = self.root, 0, []
        while offset < len(tokens):
            child = node.children.get(tokens[offset])
            if child is None:
                break
            common = self._common(child.token_sequence, tokens[offset:])
            segments.append(slice_kv(child.kv_cache, 0, common))
            offset += common
            if common < len(child.token_sequence):
                break
            node = child
        return offset, segments

    def match(self, tokens):
        """Return (matched_token_count, independent layer-wise prefix KV)."""
        matched, segments = self._walk(list(tokens))
        return matched, concatenate_kv_caches(segments)

    def insert(self, tokens, kv_cache):
        """Insert a complete prompt cache, splitting compressed nodes as needed.

        KV must correspond to exactly these tokens, at positions starting at 0.
        A token budget bounds retained tensor storage. On capacity exhaustion,
        the tree is cleared; this deliberately simple policy is not LRU.
        """
        tokens = list(tokens)
        if not tokens:
            return
        if kv_cache is None or len(kv_cache) == 0:
            raise ValueError("A nonempty prompt requires a populated KV cache")
        for k, v in kv_cache:
            if (k.ndim != 4 or k.shape != v.shape or k.size(0) != 1
                    or k.size(2) != len(tokens)):
                raise ValueError("Expected batch-one KV tensors for the entire prompt")
        if self.max_cached_tokens is not None and len(tokens) > self.max_cached_tokens:
            return  # A single oversized prompt is not retained.
        matched, _ = self._walk(tokens)
        additional = len(tokens) - matched
        if (self.max_cached_tokens is not None and
                self.cached_tokens + additional > self.max_cached_tokens):
            self.clear()
            additional = len(tokens)

        node, offset = self.root, 0
        while offset < len(tokens):
            child = node.children.get(tokens[offset])
            if child is None:
                node.children[tokens[offset]] = RadixTreeNode(
                    tokens[offset:], slice_kv(kv_cache, offset, len(tokens), copy=True)
                )
                break
            common = self._common(child.token_sequence, tokens[offset:])
            if common < len(child.token_sequence):
                # Both new segments own independent storage, including when the
                # newly inserted prompt ends inside an existing compressed node.
                prefix = RadixTreeNode(
                    child.token_sequence[:common],
                    slice_kv(child.kv_cache, 0, common, copy=True),
                )
                suffix = RadixTreeNode(
                    child.token_sequence[common:],
                    slice_kv(child.kv_cache, common, len(child.token_sequence), copy=True),
                    child.children,
                )
                prefix.children[suffix.token_sequence[0]] = suffix
                node.children[tokens[offset]] = prefix
                node = prefix
            else:
                node = child
            offset += common
        self.cached_tokens += additional

    def get_stats(self):
        nodes, nbytes = 0, 0
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            nodes += 1
            nbytes += sum(t.numel() * t.element_size() for pair in node.kv_cache for t in pair)
            stack.extend(node.children.values())
        return {"total_nodes": nodes, "cached_nodes": nodes,
                "cached_tokens": self.cached_tokens, "kv_bytes": nbytes}
