# %load radix_tree.py
"""
Radix Tree implementation for storing and reusing KV caches across prompts with shared prefixes.
"""
import torch

class RadixTreeNode:
    def __init__(self, token_sequence=None):
        self.token_sequence = token_sequence if token_sequence is not None else []
        self.children = {}  # Maps first token of child sequence -> child node
        self.kv_cache = None  # KV cache for this node's token sequence
        
    def __repr__(self):
        return f"Node(tokens={self.token_sequence}, children={len(self.children)})"


class RadixTree:
    def __init__(self):
        self.root = RadixTreeNode([])
        
    def find_longest_prefix(self, tokens):
        """
        Find the longest prefix match in the tree.
        
        Returns:
            matched_tokens: List of tokens that were matched
            matched_kv_list: List of KV caches corresponding to matched nodes
            remaining_tokens: Tokens that weren't matched
            parent_node: Node where remaining tokens should be inserted
            matched_prexfix_in_parent: How many tokens matched in the last partial node
        """
        if not tokens:
            return [], [], [], self.root, 0
            
        current_node = self.root
        matched_tokens = []
        matched_kv_list = []
        token_idx = 0
        
        while token_idx < len(tokens):
            remaining = tokens[token_idx:]
            found_child = False
            
            # Look for a child that matches
            for first_token, child in current_node.children.items():
                if first_token == remaining[0]:
                    # Found a potential match
                    child_tokens = child.token_sequence
                    
                    # Check how many tokens match
                    match_len = 0
                    for i in range(min(len(child_tokens), len(remaining))):
                        if child_tokens[i] == remaining[i]:
                            match_len += 1
                        else:
                            break
                    
                    if match_len > 0:
                        if match_len == len(child_tokens):
                            # Full match of this child node
                            matched_tokens.extend(child_tokens)
                            if child.kv_cache is not None:
                                matched_kv_list.append(child.kv_cache)
                            token_idx += match_len
                            current_node = child
                            found_child = True
                            break
                        else:
                            # Partial match - need to split this node later
                            # For now, return what we've matched so far
                            partial_match = child_tokens[:match_len]
                            matched_tokens.extend(partial_match)
                            # We can't use KV cache from partially matched node
                            remaining_tokens = tokens[token_idx + match_len:]
                            return matched_tokens, matched_kv_list, remaining_tokens, current_node, match_len
            
            if not found_child:
                # No matching child found
                break
        
        remaining_tokens = tokens[token_idx:]
        return matched_tokens, matched_kv_list, remaining_tokens, current_node, 0
    
    def insert(self, tokens, kv_cache):
        """
        Insert a token sequence and its KV cache into the tree.
        
        Args:
            tokens: List of token IDs
            kv_cache: List of (k, v) tuples for each layer
        """
        if not tokens:
            return
            
        matched_tokens, matched_kv_list, remaining_tokens, parent_node, partial_match_len = \
            self.find_longest_prefix(tokens)
        
        if not remaining_tokens:
            # Exact match already exists - update KV cache
            return
        
        if partial_match_len > 0:
            # Need to split an existing node
            first_token = remaining_tokens[0]
            child_to_split = parent_node.children[matched_tokens[-partial_match_len]]
            
            # Split the child node
            matched_part = child_to_split.token_sequence[:partial_match_len]
            remaining_part = child_to_split.token_sequence[partial_match_len:]
            
            # Create new intermediate node for matched part
            intermediate_node = RadixTreeNode(matched_part)
            intermediate_node.kv_cache = None  # We don't have KV for partial sequences
            
            # Update the child to only contain remaining part
            old_children = child_to_split.children
            old_kv = child_to_split.kv_cache
            
            child_to_split.token_sequence = remaining_part
            # Keep its children and KV cache
            
            # Insert intermediate node
            parent_node.children[matched_part[0]] = intermediate_node
            intermediate_node.children[remaining_part[0]] = child_to_split
            
            # Now insert the new remaining tokens
            parent_node = intermediate_node
            remaining_tokens = tokens[len(matched_tokens):]
        
        # Insert remaining tokens as new node(s)
        if remaining_tokens:
            # Extract KV cache only for the new tokens
            # This is tricky - we need to slice the KV cache appropriately
            prefix_len = len(matched_tokens)
            
            # Create new node
            new_node = RadixTreeNode(remaining_tokens)
            
            # Store KV cache for the new suffix
            # The kv_cache passed in contains KV for ALL tokens (prefix + suffix)
            # We need to extract only the suffix part
            if kv_cache is not None and prefix_len < len(tokens):
                new_node.kv_cache = self._extract_suffix_kv(kv_cache, prefix_len, len(tokens))
            
            parent_node.children[remaining_tokens[0]] = new_node
    
    def _extract_suffix_kv(self, full_kv_cache, start_idx, end_idx):
        """
        Extract KV cache for tokens from start_idx to end_idx.
        
        Args:
            full_kv_cache: List of (k, v) tuples for each layer
            start_idx: Starting token index
            end_idx: Ending token index (exclusive)
        
        Returns:
            Extracted KV cache for the specified range
        """
        if full_kv_cache is None:
            return None
            
        extracted_kv = []
        for k, v in full_kv_cache:
            # k and v have shape (B, n_head, seq_len, head_dim)
            # We want to extract seq_len dimension from start_idx to end_idx
            k_suffix = k[:, :, start_idx:end_idx, :]
            v_suffix = v[:, :, start_idx:end_idx, :]
            extracted_kv.append((k_suffix, v_suffix))
        
        return extracted_kv
    
    def concatenate_kv_caches(self, kv_cache_list):
        """
        Concatenate multiple KV caches along the sequence dimension.
        
        Args:
            kv_cache_list: List of KV caches, each is a list of (k, v) tuples
        
        Returns:
            Concatenated KV cache
        """
        if not kv_cache_list:
            return None
        
        if len(kv_cache_list) == 1:
            return kv_cache_list[0]
        
        num_layers = len(kv_cache_list[0])
        concatenated_kv = []
        
        for layer_idx in range(num_layers):
            k_list = [kv[layer_idx][0] for kv in kv_cache_list]
            v_list = [kv[layer_idx][1] for kv in kv_cache_list]
            
            k_concat = torch.cat(k_list, dim=2)  # Concatenate along sequence dimension
            v_concat = torch.cat(v_list, dim=2)
            
            concatenated_kv.append((k_concat, v_concat))
        
        return concatenated_kv
    
    def get_stats(self):
        """Get statistics about the tree."""
        def count_nodes(node):
            count = 1
            for child in node.children.values():
                count += count_nodes(child)
            return count
        
        def count_cached_nodes(node):
            count = 1 if node.kv_cache is not None else 0
            for child in node.children.values():
                count += count_cached_nodes(child)
            return count
        
        total_nodes = count_nodes(self.root) - 1  # Exclude root
        cached_nodes = count_cached_nodes(self.root)
        
        return {
            'total_nodes': total_nodes,
            'cached_nodes': cached_nodes
        }
    
    def print_tree(self, node=None, prefix="", is_last=True):
        """Print tree structure for debugging."""
        if node is None:
            node = self.root
            print("ROOT")
        
        children = list(node.children.values())
        for i, child in enumerate(children):
            is_last_child = (i == len(children) - 1)
            
            connector = "└── " if is_last_child else "├── "
            print(f"{prefix}{connector}{child.token_sequence} (cached: {child.kv_cache is not None})")
            
            extension = "    " if is_last_child else "│   "
            self.print_tree(child, prefix + extension, is_last_child)
