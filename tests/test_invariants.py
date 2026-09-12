"""Step-wise capacity invariant and adversarial stress tests for SynapseTieredCache."""

import unittest
import numpy as np

from engine.tiered_cache import SynapseTieredCache, CacheTier


class TestCacheInvariants(unittest.TestCase):
    """Enforces the fundamental theorem: |C_t| <= B for all steps t in [1, N]."""

    def setUp(self):
        np.random.seed(42)
        self.d_head = 32
        self.num_kv_heads = 4

    def _dummy_kv(self):
        k = np.random.randn(self.num_kv_heads, self.d_head).astype(np.float32)
        v = np.random.randn(self.num_kv_heads, self.d_head).astype(np.float32)
        return k, v

    def test_stepwise_invariant_enforcement_long_stream(self):
        """Verifies |C_t| <= B at every single insertion step across N = 5,000 tokens."""
        B = 64
        cache = SynapseTieredCache(total_budget=B, k_sink=4, w_local=16, k_anchor=32, d_head=self.d_head, num_kv_heads=self.num_kv_heads)
        
        N = 5000
        for t in range(N):
            k, v = self._dummy_kv()
            cache.insert_token(token_id=t, key=k, value=v, step=t, initial_salience=float(t % 10))
            
            # Strict Step-wise Invariant Assertion: Must hold after every single insertion
            self.assertLessEqual(cache.current_size, B, f"Invariant violated at step {t}: size={cache.current_size} > B={B}")
            
        self.assertEqual(cache.current_size, B)
        self.assertEqual(cache.peak_size_observed, B)

    def test_sink_tokens_are_strictly_never_evicted(self):
        """Sinks {0, 1, ..., k_sink-1} must remain permanently in cache."""
        B = 32
        k_sink = 4
        cache = SynapseTieredCache(total_budget=B, k_sink=k_sink, w_local=8, k_anchor=16, d_head=self.d_head, num_kv_heads=self.num_kv_heads)
        
        N = 2000
        for t in range(N):
            k, v = self._dummy_kv()
            cache.insert_token(token_id=t, key=k, value=v, step=t)
            
        for sink_id in range(k_sink):
            self.assertTrue(cache.contains(sink_id), f"Sink token {sink_id} was improperly evicted!")

    def test_adversarial_all_tokens_qualify_as_anchors(self):
        """Stress Case 1: Every single token attempts to become an anchor."""
        B = 40
        cache = SynapseTieredCache(total_budget=B, k_sink=4, w_local=8, k_anchor=20, d_head=self.d_head, num_kv_heads=self.num_kv_heads)
        
        N = 1000
        for t in range(N):
            k, v = self._dummy_kv()
            cache.insert_token(token_id=t, key=k, value=v, step=t)
            # Adversarial promotion: every token promoted
            cache.promote_to_anchor(t, salience_score=float(t))
            self.assertLessEqual(cache.current_size, B)
            
        dist = cache.get_tier_distribution()
        self.assertLessEqual(dist["anchor"], cache.k_anchor)
        self.assertLessEqual(dist["total"], B)

    def test_adversarial_zero_anchors_promoted(self):
        """Stress Case 2: No token ever qualifies as an anchor (pure candidate/local eviction)."""
        B = 50
        cache = SynapseTieredCache(total_budget=B, k_sink=5, w_local=15, k_anchor=20, d_head=self.d_head, num_kv_heads=self.num_kv_heads)
        
        N = 1000
        for t in range(N):
            k, v = self._dummy_kv()
            cache.insert_token(token_id=t, key=k, value=v, step=t)
            self.assertLessEqual(cache.current_size, B)
            
        dist = cache.get_tier_distribution()
        self.assertEqual(dist["anchor"], 0)
        self.assertEqual(dist["total"], B)

    def test_adversarial_simultaneous_tier_saturation(self):
        """Stress Case 3: Sinks, Anchors, Local, and Candidate tiers all saturated simultaneously."""
        B = 60
        cache = SynapseTieredCache(total_budget=B, k_sink=4, w_local=16, k_anchor=20, d_head=self.d_head, num_kv_heads=self.num_kv_heads)
        
        # Fill cache and promote precisely k_anchor items
        for t in range(200):
            k, v = self._dummy_kv()
            cache.insert_token(token_id=t, key=k, value=v, step=t)
            if 10 <= t < 30:
                cache.promote_to_anchor(t, salience_score=50.0 + t)
            self.assertLessEqual(cache.current_size, B)
            
        self.assertEqual(cache.current_size, B)

    def test_adversarial_repeated_anchor_re_promotion(self):
        """Stress Case 4: Repeated queries continually boost existing anchors."""
        B = 32
        cache = SynapseTieredCache(total_budget=B, k_sink=2, w_local=10, k_anchor=10, d_head=self.d_head, num_kv_heads=self.num_kv_heads)
        
        # Insert 10 initial tokens and promote them
        for t in range(10):
            k, v = self._dummy_kv()
            cache.insert_token(t, k, v, step=t)
            cache.promote_to_anchor(t, salience_score=100.0)
            
        # Stream 500 new tokens while re-promoting token 5
        for t in range(10, 500):
            k, v = self._dummy_kv()
            cache.insert_token(t, k, v, step=t)
            if t % 5 == 0:
                cache.promote_to_anchor(5, salience_score=200.0 + t)
            self.assertLessEqual(cache.current_size, B)
            
        # Token 5 must be preserved
        self.assertTrue(cache.contains(5))

    def test_adversarial_ultra_constrained_budgets(self):
        """Stress Case 6: Extremely small budgets B in [8, 16, 24]."""
        for B in [8, 16, 24]:
            cache = SynapseTieredCache(total_budget=B, k_sink=2, w_local=4, k_anchor=4, d_head=self.d_head, num_kv_heads=self.num_kv_heads)
            for t in range(200):
                k, v = self._dummy_kv()
                cache.insert_token(t, k, v, step=t)
                if t % 3 == 0:
                    cache.promote_to_anchor(t, salience_score=float(t))
                self.assertLessEqual(cache.current_size, B)
            self.assertEqual(cache.current_size, B)

    def test_adversarial_sub_minimal_budget_stress(self):
        """Stress Case 7: Requested B <= k_sink + w_local."""
        # User requests B = 6 with k_sink=4, w_local=16. The cache must automatically rescale safely.
        cache = SynapseTieredCache(total_budget=6, k_sink=4, w_local=16, k_anchor=10, d_head=self.d_head, num_kv_heads=self.num_kv_heads)
        self.assertLessEqual(cache.B, 6)
        
        for t in range(100):
            k, v = self._dummy_kv()
            cache.insert_token(t, k, v, step=t)
            self.assertLessEqual(cache.current_size, 6)
        self.assertEqual(cache.current_size, 6)


if __name__ == "__main__":
    unittest.main()
