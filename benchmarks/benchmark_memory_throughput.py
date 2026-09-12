"""Memory Footprint and Forward-Step Latency Profiling for SynapseCache."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import numpy as np
from typing import Dict, List, Any

from engine.tiered_cache import SynapseTieredCache
from engine.transformer_attention import MultiHeadAttentionSimulator, AttentionConfig
from baselines.baselines import FullKVCache, SlidingWindowCache, StreamingLLMCache, H2OHeavyHitterCache


def compute_kv_memory_mb(
    active_tokens: int,
    L: int = 80,
    H_KV: int = 8,
    d_head: int = 128,
    b: int = 2
) -> float:
    """Computes exact KV cache memory in Megabytes:
    
    Memory = 2 * active_tokens * L * H_KV * d_head * b (bytes) / (1024 * 1024)
    """
    total_bytes = 2 * active_tokens * L * H_KV * d_head * b
    return total_bytes / (1024.0 * 1024.0)


def run_memory_throughput_benchmark():
    print("=" * 70)
    print("SYNAPSECACHE EMPIRICAL BENCHMARK: MEMORY & LATENCY PROFILING")
    print("=" * 70)
    
    sequence_lengths = [2000, 4000, 8000, 16000, 32000, 64000]
    budget_ratio = 0.20  # 80% theoretical reduction
    
    config = AttentionConfig(d_model=512, num_q_heads=8, num_kv_heads=8, d_head=64)
    simulator = MultiHeadAttentionSimulator(config)
    
    results: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_reference": "70B Parameter Architecture (L=80, H_KV=8, d_head=128, FP16)",
            "budget_ratio": budget_ratio
        },
        "scaling_data": []
    }
    
    for N in sequence_lengths:
        B = int(N * budget_ratio)
        
        full_cache = FullKVCache(d_head=64, num_kv_heads=8)
        synapse_cache = SynapseTieredCache(total_budget=B, k_sink=4, w_local=B // 4, k_anchor=B // 2, d_head=64, num_kv_heads=8)
        
        # Populate caches to simulate state at step N
        np.random.seed(42)
        # Fast population
        for t in range(N):
            k = np.random.randn(8, 64).astype(np.float32)
            v = np.random.randn(8, 64).astype(np.float32)
            full_cache.insert_token(t, k, v, step=t)
            synapse_cache.insert_token(t, k, v, step=t)
            if t % 50 == 0:
                synapse_cache.promote_to_anchor(t, salience_score=float(t))
                
        # Measure Attention Step Latency
        q_t = np.random.randn(8, 64).astype(np.float32)
        
        # 1. Full KV Latency
        f_ids, f_k, f_v = full_cache.get_keys_and_values()
        latencies_full = []
        for _ in range(5):
            out_full = simulator.forward_step(q_t, f_ids, f_k, f_v)
            latencies_full.append(out_full.execution_time_ms)
        mean_lat_full = float(np.median(latencies_full))
        
        # 2. SynapseCache Latency
        s_ids, s_k, s_v = synapse_cache.get_keys_and_values()
        latencies_syn = []
        for _ in range(5):
            out_syn = simulator.forward_step(q_t, s_ids, s_k, s_v)
            latencies_syn.append(out_syn.execution_time_ms)
        mean_lat_syn = float(np.median(latencies_syn))
        
        # Exact Memory Footprints
        full_mem_mb = compute_kv_memory_mb(full_cache.current_size)
        syn_mem_mb = compute_kv_memory_mb(synapse_cache.current_size)
        mem_reduction_pct = ((full_mem_mb - syn_mem_mb) / full_mem_mb) * 100.0
        speedup = mean_lat_full / max(1e-5, mean_lat_syn)
        
        row = {
            "sequence_length": N,
            "cache_budget": B,
            "full_kv_entries": full_cache.current_size,
            "synapse_entries": synapse_cache.current_size,
            "full_kv_memory_mb": round(full_mem_mb, 2),
            "synapse_memory_mb": round(syn_mem_mb, 2),
            "memory_reduction_pct": round(mem_reduction_pct, 2),
            "full_kv_latency_ms": round(mean_lat_full, 3),
            "synapse_latency_ms": round(mean_lat_syn, 3),
            "step_speedup": round(speedup, 2)
        }
        
        print(f"[N={N:>5}] Full KV: {full_mem_mb:8.2f} MB ({mean_lat_full:6.2f}ms) | Synapse: {syn_mem_mb:8.2f} MB ({mean_lat_syn:6.2f}ms) | Reduc: {mem_reduction_pct:5.1f}% | Speedup: {speedup:4.2f}x")
        results["scaling_data"].append(row)
        
    out_dir = r"c:\Users\rswar\OneDrive\Desktop\SynapseCache\data\benchmark_results"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "memory_latency_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 70)
    print(f"Memory & Latency Profiling saved to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_memory_throughput_benchmark()
