"""Comparative Baseline KV-Caches for SynapseCache Evaluation.

Includes:
1. FullKVCache: Exact reference ceiling with unbounded O(N) memory.
2. SlidingWindowCache: Pure recency cache retaining only the most recent B tokens.
3. StreamingLLMCache: Sinks (k_sink) + local sliding window (B - k_sink).
4. H2OHeavyHitterCache: Greedy historical attention mass accumulator + local window.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np


class FullKVCache:
    """Exact unconstrained reference ceiling: Retains all tokens O(N)."""
    
    def __init__(self, d_head: int = 64, num_kv_heads: int = 8):
        self.d_head = d_head
        self.num_kv_heads = num_kv_heads
        self.keys: Dict[int, np.ndarray] = {}
        self.values: Dict[int, np.ndarray] = {}
        
    @property
    def current_size(self) -> int:
        return len(self.keys)
        
    def contains(self, token_id: int) -> bool:
        return token_id in self.keys
        
    def get_keys_and_values(self) -> Tuple[List[int], np.ndarray, np.ndarray]:
        sorted_ids = sorted(self.keys.keys())
        if not sorted_ids:
            return sorted_ids, np.zeros((0, self.num_kv_heads, self.d_head), dtype=np.float32), np.zeros((0, self.num_kv_heads, self.d_head), dtype=np.float32)
        k_list = [self.keys[tid] for tid in sorted_ids]
        v_list = [self.values[tid] for tid in sorted_ids]
        return sorted_ids, np.stack(k_list, axis=0), np.stack(v_list, axis=0)
        
    def insert_token(self, token_id: int, key: np.ndarray, value: np.ndarray, step: int, **kwargs) -> Optional[int]:
        self.keys[token_id] = key
        self.values[token_id] = value
        return None
        
    def update_step(self, step_attentions: Dict[int, List[float]]) -> None:
        pass


class SlidingWindowCache:
    """Pure recency baseline: Retains only the most recent B tokens."""
    
    def __init__(self, total_budget: int = 128, d_head: int = 64, num_kv_heads: int = 8):
        self.B = max(1, total_budget)
        self.d_head = d_head
        self.num_kv_heads = num_kv_heads
        self.keys: Dict[int, np.ndarray] = {}
        self.values: Dict[int, np.ndarray] = {}
        self.order: List[int] = []
        
    @property
    def current_size(self) -> int:
        return len(self.keys)
        
    def contains(self, token_id: int) -> bool:
        return token_id in self.keys
        
    def get_keys_and_values(self) -> Tuple[List[int], np.ndarray, np.ndarray]:
        sorted_ids = sorted(self.keys.keys())
        if not sorted_ids:
            return sorted_ids, np.zeros((0, self.num_kv_heads, self.d_head), dtype=np.float32), np.zeros((0, self.num_kv_heads, self.d_head), dtype=np.float32)
        k_list = [self.keys[tid] for tid in sorted_ids]
        v_list = [self.values[tid] for tid in sorted_ids]
        return sorted_ids, np.stack(k_list, axis=0), np.stack(v_list, axis=0)
        
    def insert_token(self, token_id: int, key: np.ndarray, value: np.ndarray, step: int, **kwargs) -> Optional[int]:
        evicted = None
        self.keys[token_id] = key
        self.values[token_id] = value
        self.order.append(token_id)
        
        if len(self.keys) > self.B:
            evicted = self.order.pop(0)
            self.keys.pop(evicted, None)
            self.values.pop(evicted, None)
            
        assert len(self.keys) <= self.B
        return evicted
        
    def update_step(self, step_attentions: Dict[int, List[float]]) -> None:
        pass


class StreamingLLMCache:
    """StreamingLLM baseline: Retains k_sink initial tokens + (B - k_sink) recent tokens."""
    
    def __init__(self, total_budget: int = 128, k_sink: int = 4, d_head: int = 64, num_kv_heads: int = 8):
        self.B = max(4, total_budget)
        self.k_sink = min(k_sink, self.B // 4)
        self.w_local = self.B - self.k_sink
        self.d_head = d_head
        self.num_kv_heads = num_kv_heads
        
        self.keys: Dict[int, np.ndarray] = {}
        self.values: Dict[int, np.ndarray] = {}
        self.sink_ids: Set[int] = set()
        self.local_order: List[int] = []
        
    @property
    def current_size(self) -> int:
        return len(self.keys)
        
    def contains(self, token_id: int) -> bool:
        return token_id in self.keys
        
    def get_keys_and_values(self) -> Tuple[List[int], np.ndarray, np.ndarray]:
        sorted_ids = sorted(self.keys.keys())
        if not sorted_ids:
            return sorted_ids, np.zeros((0, self.num_kv_heads, self.d_head), dtype=np.float32), np.zeros((0, self.num_kv_heads, self.d_head), dtype=np.float32)
        k_list = [self.keys[tid] for tid in sorted_ids]
        v_list = [self.values[tid] for tid in sorted_ids]
        return sorted_ids, np.stack(k_list, axis=0), np.stack(v_list, axis=0)
        
    def insert_token(self, token_id: int, key: np.ndarray, value: np.ndarray, step: int, **kwargs) -> Optional[int]:
        self.keys[token_id] = key
        self.values[token_id] = value
        
        if token_id < self.k_sink:
            self.sink_ids.add(token_id)
        else:
            self.local_order.append(token_id)
            
        evicted = None
        if len(self.local_order) > self.w_local:
            evicted = self.local_order.pop(0)
            self.keys.pop(evicted, None)
            self.values.pop(evicted, None)
            
        assert len(self.keys) <= self.B
        return evicted
        
    def update_step(self, step_attentions: Dict[int, List[float]]) -> None:
        pass


class H2OHeavyHitterCache:
    """H2O baseline: Sinks + Top-k accumulated attention heavy hitters + local window."""
    
    def __init__(
        self,
        total_budget: int = 128,
        k_sink: int = 4,
        w_local: int = 32,
        d_head: int = 64,
        num_kv_heads: int = 8
    ):
        self.B = max(4, total_budget)
        self.k_sink = min(k_sink, self.B // 4)
        self.w_local = min(w_local, (self.B - self.k_sink) // 2)
        self.k_heavy = self.B - self.k_sink - self.w_local
        
        self.d_head = d_head
        self.num_kv_heads = num_kv_heads
        
        self.keys: Dict[int, np.ndarray] = {}
        self.values: Dict[int, np.ndarray] = {}
        self.sink_ids: Set[int] = set()
        self.local_ids: List[int] = []
        self.heavy_ids: Set[int] = set()
        
        self.accumulated_attention: Dict[int, float] = {}
        
    @property
    def current_size(self) -> int:
        return len(self.keys)
        
    def contains(self, token_id: int) -> bool:
        return token_id in self.keys
        
    def get_keys_and_values(self) -> Tuple[List[int], np.ndarray, np.ndarray]:
        sorted_ids = sorted(self.keys.keys())
        if not sorted_ids:
            return sorted_ids, np.zeros((0, self.num_kv_heads, self.d_head), dtype=np.float32), np.zeros((0, self.num_kv_heads, self.d_head), dtype=np.float32)
        k_list = [self.keys[tid] for tid in sorted_ids]
        v_list = [self.values[tid] for tid in sorted_ids]
        return sorted_ids, np.stack(k_list, axis=0), np.stack(v_list, axis=0)
        
    def insert_token(self, token_id: int, key: np.ndarray, value: np.ndarray, step: int, **kwargs) -> Optional[int]:
        self.keys[token_id] = key
        self.values[token_id] = value
        self.accumulated_attention[token_id] = 0.0
        
        if token_id < self.k_sink:
            self.sink_ids.add(token_id)
        else:
            self.local_ids.append(token_id)
            
        # If local window overflows, demote oldest to heavy hitter candidate pool
        if len(self.local_ids) > self.w_local:
            demoted = self.local_ids.pop(0)
            self.heavy_ids.add(demoted)
            
        evicted = None
        # If heavy hitters exceed k_heavy, evict token with lowest accumulated attention
        if len(self.heavy_ids) > self.k_heavy:
            evicted = min(self.heavy_ids, key=lambda tid: self.accumulated_attention.get(tid, 0.0))
            self.heavy_ids.remove(evicted)
            self.keys.pop(evicted, None)
            self.values.pop(evicted, None)
            self.accumulated_attention.pop(evicted, None)
            
        assert len(self.keys) <= self.B
        return evicted
        
    def update_step(self, step_attentions: Dict[int, List[float]]) -> None:
        """Accumulates mean attention mass across heads for each active token."""
        for tid, head_weights in step_attentions.items():
            if tid in self.keys:
                mean_mass = sum(head_weights) / len(head_weights) if head_weights else 0.0
                self.accumulated_attention[tid] = self.accumulated_attention.get(tid, 0.0) + mean_mass
