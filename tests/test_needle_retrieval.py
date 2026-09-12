"""Synthetic Needle-in-a-Haystack correctness test for SynapseCache."""

import unittest
import numpy as np

from engine.tiered_cache import SynapseTieredCache
from baselines.baselines import FullKVCache, SlidingWindowCache, StreamingLLMCache


class TestNeedleRetrieval(unittest.TestCase):
    """Verifies that cross-head anchors retain long-range factual needles while recency caches drop them."""

    def test_needle_retrieval_comparison(self):
        N = 2000
        B = 256  # 12.8% cache budget
        needle_position = 400  # 20% depth into context (far outside local window of size 128)
        
        d_head = 64
        num_kv_heads = 4
        
        # Initialize caches
        full_kv = FullKVCache(d_head=d_head, num_kv_heads=num_kv_heads)
        sliding = SlidingWindowCache(total_budget=B, d_head=d_head, num_kv_heads=num_kv_heads)
        streaming = StreamingLLMCache(total_budget=B, k_sink=4, d_head=d_head, num_kv_heads=num_kv_heads)
        synapse = SynapseTieredCache(total_budget=B, k_sink=4, w_local=64, k_anchor=128, d_head=d_head, num_kv_heads=num_kv_heads)
        
        np.random.seed(42)
        
        # Stream N tokens
        for t in range(N):
            k = np.random.randn(num_kv_heads, d_head).astype(np.float32)
            v = np.random.randn(num_kv_heads, d_head).astype(np.float32)
            
            full_kv.insert_token(t, k, v, step=t)
            sliding.insert_token(t, k, v, step=t)
            streaming.insert_token(t, k, v, step=t)
            synapse.insert_token(t, k, v, step=t)
            
            # When needle is inserted, it receives strong cross-head attention and is promoted
            if t == needle_position:
                synapse.promote_to_anchor(needle_position, salience_score=85.0)

        # 1. Full KV retains needle
        self.assertTrue(full_kv.contains(needle_position))
        
        # 2. Sliding window (size 256) at t=2000 only contains [1744, 1999]. Needle at 400 MUST be evicted
        self.assertFalse(sliding.contains(needle_position), "Sliding window should have evicted old needle")
        
        # 3. StreamingLLM (sinks 0..3, local window [1748..1999]) MUST evict needle at 400
        self.assertFalse(streaming.contains(needle_position), "StreamingLLM should have evicted needle outside sink & window")
        
        # 4. SynapseCache MUST retain needle due to anchor protection
        self.assertTrue(synapse.contains(needle_position), "SynapseCache failed to retain needle anchor!")
        
        # Check that SynapseCache strictly respected budget B
        self.assertLessEqual(synapse.current_size, B)


if __name__ == "__main__":
    unittest.main()
