"""SynapseCache Engine: Bounded-Memory Attention-Entropy KV-Cache for Long-Context Transformers."""

from .attention_entropy import (
    compute_cross_head_distribution,
    compute_cross_head_entropy,
    compute_mean_attention_mass,
    evaluate_anchor_criterion,
    compute_token_salience,
    AttentionScorer,
    SalienceScores
)
from .tiered_cache import SynapseTieredCache, CacheTier, RetainedToken
from .transformer_attention import (
    MultiHeadAttentionSimulator,
    AttentionConfig,
    AttentionOutput
)

__all__ = [
    "compute_cross_head_distribution",
    "compute_cross_head_entropy",
    "compute_mean_attention_mass",
    "evaluate_anchor_criterion",
    "compute_token_salience",
    "AttentionScorer",
    "SalienceScores",
    "SynapseTieredCache",
    "CacheTier",
    "RetainedToken",
    "MultiHeadAttentionSimulator",
    "AttentionConfig",
    "AttentionOutput"
]
