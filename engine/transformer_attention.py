"""Multi-Head & Grouped-Query Attention Simulator for SynapseCache.

Implements scaled dot-product attention with configurable query heads (H_Q)
and KV heads (H_KV), computing attention output and cross-head weight matrices.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple, Any
import math
import time
import numpy as np


class AttentionConfig:
    """Configuration hyper-parameters for multi-head attention."""
    
    def __init__(
        self,
        d_model: int = 512,
        num_q_heads: int = 8,
        num_kv_heads: int = 8,
        d_head: int = 64
    ):
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.d_head = d_head
        self.scale = 1.0 / math.sqrt(d_head)
        
        # In GQA, each KV head is shared by (H_Q // H_KV) query heads
        assert num_q_heads % num_kv_heads == 0, "H_Q must be divisible by H_KV"
        self.heads_per_kv = num_q_heads // num_kv_heads


class AttentionOutput(NamedTuple):
    """Output bundle from an attention generation step."""
    output: np.ndarray             # Shape: (d_model,)
    step_attentions: Dict[int, List[float]]  # token_id -> List[weight_head_1, ..., weight_head_H]
    execution_time_ms: float
    cache_size: int


class MultiHeadAttentionSimulator:
    """Simulates autoregressive Multi-Head / Grouped-Query Attention."""

    def __init__(self, config: Optional[AttentionConfig] = None):
        self.config = config or AttentionConfig()
        
        # Projection weight matrices for reproducibility
        np.random.seed(42)
        H_Q = self.config.num_q_heads
        H_KV = self.config.num_kv_heads
        d_h = self.config.d_head
        d_m = self.config.d_model
        
        self.W_q = np.random.randn(d_m, H_Q, d_h).astype(np.float32) * 0.02
        self.W_k = np.random.randn(d_m, H_KV, d_h).astype(np.float32) * 0.02
        self.W_v = np.random.randn(d_m, H_KV, d_h).astype(np.float32) * 0.02
        self.W_o = np.random.randn(H_Q * d_h, d_m).astype(np.float32) * 0.02

    def project_token(
        self,
        x_t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Projects input embedding x_t into (Query, Key, Value) vectors.
        
        Args:
            x_t: Input token embedding vector of shape (d_model,)
            
        Returns:
            q: np.ndarray of shape (H_Q, d_head)
            k: np.ndarray of shape (H_KV, d_head)
            v: np.ndarray of shape (H_KV, d_head)
        """
        q = np.einsum("m,mhd->hd", x_t, self.W_q)
        k = np.einsum("m,mhd->hd", x_t, self.W_k)
        v = np.einsum("m,mhd->hd", x_t, self.W_v)
        return q, k, v

    def forward_step(
        self,
        q_t: np.ndarray,
        active_token_ids: List[int],
        cached_keys: np.ndarray,
        cached_values: np.ndarray
    ) -> AttentionOutput:
        """Computes scaled dot-product attention against active KV cache entries.
        
        Args:
            q_t: Query vector for current step, shape (H_Q, d_head)
            active_token_ids: List of active token IDs in cache of length S
            cached_keys: Cached Key tensor of shape (S, H_KV, d_head)
            cached_values: Cached Value tensor of shape (S, H_KV, d_head)
            
        Returns:
            AttentionOutput containing output vector, cross-head attention weights, latency.
        """
        t0 = time.perf_counter()
        
        S = len(active_token_ids)
        H_Q = self.config.num_q_heads
        H_KV = self.config.num_kv_heads
        d_h = self.config.d_head
        
        if S == 0:
            out = np.zeros(self.config.d_model, dtype=np.float32)
            t1 = time.perf_counter()
            return AttentionOutput(
                output=out,
                step_attentions={},
                execution_time_ms=(t1 - t0) * 1000.0,
                cache_size=0
            )

        # Broadcast KV heads to match query heads for GQA
        # q_t shape: (H_Q, d_h)
        # cached_keys shape: (S, H_KV, d_h)
        # Expand cached_keys across query heads:
        k_expanded = np.repeat(cached_keys, self.config.heads_per_kv, axis=1)  # (S, H_Q, d_h)
        v_expanded = np.repeat(cached_values, self.config.heads_per_kv, axis=1)  # (S, H_Q, d_h)

        # Scaled dot-product: (q * k) * scale
        # scores shape: (H_Q, S)
        scores = np.einsum("hd,shd->hs", q_t, k_expanded) * self.config.scale

        # Softmax over sequence dimension S for each head
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-12)  # (H_Q, S)

        # Context output: attention_weights * v
        # context shape: (H_Q, d_h)
        context = np.einsum("hs,shd->hd", attention_weights, v_expanded)

        # Output projection
        flattened_context = context.reshape(-1)
        final_output = np.dot(flattened_context, self.W_o)

        # Build step_attentions dict mapping token_id -> List[weight_h1, ..., weight_h_HQ]
        step_attentions: Dict[int, List[float]] = {}
        for s_idx, tid in enumerate(active_token_ids):
            step_attentions[tid] = [float(attention_weights[h, s_idx]) for h in range(H_Q)]

        t1 = time.perf_counter()
        
        return AttentionOutput(
            output=final_output,
            step_attentions=step_attentions,
            execution_time_ms=(t1 - t0) * 1000.0,
            cache_size=S
        )
