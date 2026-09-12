"""Unit tests for SynapseCache mathematical primitives."""

import unittest
import math
import numpy as np

from engine.attention_entropy import (
    compute_cross_head_distribution,
    compute_cross_head_entropy,
    compute_mean_attention_mass,
    evaluate_anchor_criterion,
    compute_token_salience,
    AttentionScorer
)


class TestAttentionPrimitives(unittest.TestCase):
    """Verifies mathematical invariants of the attention entropy module."""

    def test_cross_head_distribution_sums_to_one(self):
        """Probability distribution across heads must strictly sum to 1.0."""
        weights = [0.1, 0.4, 0.2, 0.3, 0.05, 0.15, 0.8, 0.0]
        p_dist = compute_cross_head_distribution(weights)
        self.assertEqual(len(p_dist), len(weights))
        self.assertAlmostEqual(sum(p_dist), 1.0, places=6)
        self.assertTrue(all(p >= 0.0 for p in p_dist))

    def test_cross_head_distribution_zeros(self):
        """Zero inputs must produce safe, valid non-NaN outputs."""
        weights = [0.0, 0.0, 0.0, 0.0]
        p_dist = compute_cross_head_distribution(weights)
        self.assertEqual(len(p_dist), 4)
        self.assertTrue(all(not math.isnan(p) for p in p_dist))

    def test_entropy_extremes(self):
        """Entropy must be maximal (1.0) on uniform distribution and minimal (0.0) on delta."""
        # 1. Delta distribution (all weight on head 0)
        delta_weights = [1.0, 0.0, 0.0, 0.0]
        h_delta = compute_cross_head_entropy(delta_weights)
        self.assertAlmostEqual(h_delta, 0.0, places=4)

        # 2. Perfectly uniform distribution across 4 heads
        uniform_weights = [0.25, 0.25, 0.25, 0.25]
        h_uniform = compute_cross_head_entropy(uniform_weights)
        self.assertAlmostEqual(h_uniform, 1.0, places=4)

    def test_anchor_criterion_logic(self):
        """Anchor criterion requires >= kappa heads with attention > tau_head."""
        # Heads: 8 heads. Threshold = 0.05, Kappa = 3
        # Weights with 4 heads > 0.05 -> Should be True
        w_anchor = [0.12, 0.15, 0.08, 0.20, 0.01, 0.02, 0.01, 0.01]
        self.assertTrue(evaluate_anchor_criterion(w_anchor, tau_head=0.05, kappa_heads=3))

        # Weights with only 1 head > 0.05 -> Should be False
        w_single = [0.85, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        self.assertFalse(evaluate_anchor_criterion(w_single, tau_head=0.05, kappa_heads=3))

    def test_attention_scorer_state_decay(self):
        """AttentionScorer must exponentially decay accumulated salience."""
        scorer = AttentionScorer(gamma=0.5, decay_lambda=0.5)
        step_1 = {
            10: [0.2] * 8,
            20: [0.01] * 8
        }
        res_1 = scorer.score_step(step_1)
        score_10_initial = scorer.get_accumulated_score(10)
        self.assertGreater(score_10_initial, 0.0)

        # Next step: token 10 not queried, should decay by 0.5
        step_2 = {
            30: [0.1] * 8
        }
        scorer.score_step(step_2)
        score_10_decayed = scorer.get_accumulated_score(10)
        self.assertAlmostEqual(score_10_decayed, score_10_initial * 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
