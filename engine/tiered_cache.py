"""Tiered KV-Cache Architecture for SynapseCache.

Enforces the strict capacity invariant:
    |C_t| <= B,  for all t in [1, N]
through deterministic four-tier partitioning:
    S_sink     (Fixed attention sinks, never evicted)
    S_anchor   (Cross-head relational anchors, bounded by k_anchor)
    S_local    (Sliding recency window, bounded by w_local)
    S_candidate(Eviction candidates scored by attention-entropy salience)
"""

from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np


class CacheTier(Enum):
    SINK = "sink"
    ANCHOR = "anchor"
    LOCAL = "local"
    CANDIDATE = "candidate"


class RetainedToken:
    """Representation of an active key-value entry in the cache."""
    
    def __init__(
        self,
        token_id: int,
        key: np.ndarray,
        value: np.ndarray,
        tier: CacheTier,
        step: int,
        salience: float = 0.0
    ):
        self.token_id = token_id
        self.key = key
        self.value = value
        self.tier = tier
        self.step = step
        self.salience = salience
        self.anchor_hits = 0

    def __repr__(self) -> str:
        return f"Token(id={self.token_id}, tier={self.tier.value}, step={self.step}, score={self.salience:.4f})"


class SynapseTieredCache:
    """Bounded-memory KV Cache controller enforcing |C_t| <= B for all steps."""

    def __init__(
        self,
        total_budget: int = 128,
        k_sink: int = 4,
        w_local: int = 32,
        k_anchor: int = 64,
        d_head: int = 64,
        num_kv_heads: int = 8
    ):
        """Initializes the tiered cache with strict budget allocation.
        
        Args:
            total_budget (B): Maximum allowed retained tokens across all tiers.
            k_sink: Number of initial sink tokens to permanently preserve.
            w_local: Target size of the sliding local window.
            k_anchor: Maximum allowed relational anchors.
            d_head: Head dimension for keys/values.
            num_kv_heads: Number of KV projection heads (H_KV).
        """
        self.B = max(4, int(total_budget))
        self.d_head = d_head
        self.num_kv_heads = num_kv_heads
        
        # Invariant safety: Ensure base tiers fit inside total budget B
        self.k_sink = min(k_sink, max(1, self.B // 8))
        remaining = self.B - self.k_sink
        
        # Allocate local window and anchor capacity from remaining budget
        self.w_local = min(w_local, max(1, remaining // 2))
        self.k_anchor = min(k_anchor, remaining - self.w_local)
        self.k_candidate = self.B - (self.k_sink + self.w_local + self.k_anchor)
        
        # Primary storage: token_id -> RetainedToken
        self.entries: Dict[int, RetainedToken] = {}
        
        # Partitioned index sets for fast membership checks
        self.sink_ids: Set[int] = set()
        self.anchor_ids: Set[int] = set()
        self.local_ids: List[int] = []  # Ordered queue for sliding window
        self.candidate_ids: Set[int] = set()
        
        # Telemetry counters
        self.total_tokens_seen: int = 0
        self.total_evictions: int = 0
        self.peak_size_observed: int = 0

    @property
    def current_size(self) -> int:
        """Returns current number of active KV entries."""
        return len(self.entries)

    def contains(self, token_id: int) -> bool:
        """Checks if a token is currently retained in cache."""
        return token_id in self.entries

    def get_keys_and_values(self) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """Returns sorted token IDs and their stacked (Key, Value) tensors.
        
        Returns:
            sorted_token_ids: List[int] sorted in original sequence order.
            keys: np.ndarray of shape (len, H_KV, d_head)
            values: np.ndarray of shape (len, H_KV, d_head)
        """
        sorted_ids = sorted(self.entries.keys())
        if not sorted_ids:
            empty_k = np.zeros((0, self.num_kv_heads, self.d_head), dtype=np.float32)
            empty_v = np.zeros((0, self.num_kv_heads, self.d_head), dtype=np.float32)
            return sorted_ids, empty_k, empty_v
            
        k_list = [self.entries[tid].key for tid in sorted_ids]
        v_list = [self.entries[tid].value for tid in sorted_ids]
        
        return sorted_ids, np.stack(k_list, axis=0), np.stack(v_list, axis=0)

    def insert_token(
        self,
        token_id: int,
        key: np.ndarray,
        value: np.ndarray,
        step: int,
        initial_salience: float = 0.0
    ) -> Optional[int]:
        """Inserts a new token entry and enforces capacity invariant |C_t| <= B.
        
        Returns:
            evicted_token_id: int if an eviction occurred, else None.
        """
        self.total_tokens_seen += 1
        
        # 1. Check if token qualifies for Invariant Sink Tier
        if token_id < self.k_sink:
            tier = CacheTier.SINK
            self.sink_ids.add(token_id)
        else:
            # Default new tokens start in Local Window
            tier = CacheTier.LOCAL
            self.local_ids.append(token_id)
            
        token = RetainedToken(
            token_id=token_id,
            key=key,
            value=value,
            tier=tier,
            step=step,
            salience=initial_salience
        )
        self.entries[token_id] = token
        
        # 2. Rebalance local window if it overflows w_local
        if len(self.local_ids) > self.w_local:
            oldest_local_id = self.local_ids.pop(0)
            # Demote oldest local token to candidate tier (unless it was promoted to anchor)
            if oldest_local_id in self.entries and oldest_local_id not in self.anchor_ids and oldest_local_id not in self.sink_ids:
                self.entries[oldest_local_id].tier = CacheTier.CANDIDATE
                self.candidate_ids.add(oldest_local_id)

        # 3. Enforce strict budget invariant |C_t| <= B
        evicted_id = None
        if len(self.entries) > self.B:
            evicted_id = self._evict_one()

        self.peak_size_observed = max(self.peak_size_observed, len(self.entries))
        
        # Strict invariant assertion: Must NEVER exceed B
        assert len(self.entries) <= self.B, f"Invariant violation: cache size {len(self.entries)} exceeds budget {self.B}"
        
        return evicted_id

    def promote_to_anchor(self, token_id: int, salience_score: float) -> None:
        """Promotes a qualifying token to the Relational Anchor tier.
        
        If anchor tier exceeds k_anchor capacity, evicts the anchor with lowest score.
        """
        if token_id not in self.entries or token_id in self.sink_ids:
            return
            
        token = self.entries[token_id]
        token.salience = salience_score
        token.anchor_hits += 1
        
        if token_id not in self.anchor_ids:
            # Remove from other tier indices
            if token_id in self.candidate_ids:
                self.candidate_ids.remove(token_id)
            if token_id in self.local_ids:
                self.local_ids.remove(token_id)
                
            token.tier = CacheTier.ANCHOR
            self.anchor_ids.add(token_id)
            
        # Rebalance anchors if anchor capacity exceeded
        if len(self.anchor_ids) > self.k_anchor:
            # Demote the lowest-scoring anchor to candidate tier
            min_anchor_id = min(self.anchor_ids, key=lambda tid: self.entries[tid].salience)
            self.anchor_ids.remove(min_anchor_id)
            self.entries[min_anchor_id].tier = CacheTier.CANDIDATE
            self.candidate_ids.add(min_anchor_id)

    def update_salience(self, salience_dict: Dict[int, float]) -> None:
        """Updates salience scores for active retained tokens."""
        for tid, score in salience_dict.items():
            if tid in self.entries:
                self.entries[tid].salience = score

    def _evict_one(self) -> int:
        """Evicts exactly one token according to strict priority hierarchy:
        
        Priority 1: Lowest-salience candidate token.
        Priority 2: If no candidates, demote and evict lowest-salience anchor.
        Priority 3: If no anchors, evict oldest local window token.
        Invariant: Never evict Sink tokens.
        """
        target_id: Optional[int] = None
        
        if self.candidate_ids:
            # Evict candidate with lowest accumulated salience
            target_id = min(self.candidate_ids, key=lambda tid: self.entries[tid].salience)
            self.candidate_ids.remove(target_id)
        elif self.anchor_ids:
            # Evict anchor with lowest salience
            target_id = min(self.anchor_ids, key=lambda tid: self.entries[tid].salience)
            self.anchor_ids.remove(target_id)
        elif self.local_ids:
            # Evict oldest local token
            target_id = self.local_ids.pop(0)
        else:
            raise RuntimeError("Fatal: Cache full but only sink tokens remain; budget B is too small!")
            
        del self.entries[target_id]
        self.total_evictions += 1
        return target_id

    def get_tier_distribution(self) -> Dict[str, int]:
        """Returns count of active tokens in each tier."""
        return {
            "sink": len(self.sink_ids),
            "anchor": len(self.anchor_ids),
            "local": len(self.local_ids),
            "candidate": len(self.candidate_ids),
            "total": len(self.entries)
        }
