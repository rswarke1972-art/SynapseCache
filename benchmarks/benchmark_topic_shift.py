"""Topic-Shift Recovery Benchmark for SynapseCache.

Evaluates cache retention across sharp conversational topic migrations:
Topic A (Initial Facts) -> Topic B (Technical Discourse) -> Topic C (Distractor) -> Query Topic A
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import numpy as np
from typing import Dict, List, Any

from engine.tiered_cache import SynapseTieredCache
from baselines.baselines import FullKVCache, SlidingWindowCache, StreamingLLMCache, H2OHeavyHitterCache


def run_topic_shift_benchmark():
    print("=" * 70)
    print("SYNAPSECACHE EMPIRICAL BENCHMARK: TOPIC-SHIFT RECOVERY")
    print("=" * 70)
    
    context_configs = [
        {"N": 3000, "topics": [1000, 1000, 1000], "budget": 300},   # 90% reduction
        {"N": 6000, "topics": [2000, 2000, 2000], "budget": 600},   # 90% reduction
        {"N": 12000, "topics": [4000, 4000, 4000], "budget": 1200}, # 90% reduction
    ]
    
    num_trials = 10
    models = ["full_kv", "sliding_window", "streaming_llm", "h2o", "synapse"]
    
    results: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_trials": num_trials,
            "models": models
        },
        "topic_shift_data": []
    }
    
    for cfg in context_configs:
        N = cfg["N"]
        B = cfg["budget"]
        n_a, n_b, n_c = cfg["topics"]
        
        row: Dict[str, Any] = {
            "total_tokens": N,
            "topic_partition": cfg["topics"],
            "cache_budget": B,
            "memory_reduction_pct": 90.0,
            "topic_a_recovery_rate": {}
        }
        print(f"\n[Topic-Shift Test] N={N} tokens (A={n_a}, B={n_b}, C={n_c}) | Budget B={B}")
        
        # In Topic A, inject 5 distinct relational facts at random steps
        np.random.seed(42 + N)
        topic_a_fact_steps = sorted(np.random.choice(range(50, n_a - 50), size=5, replace=False).tolist())
        
        for m in models:
            recovered_facts = 0
            total_facts = len(topic_a_fact_steps) * num_trials
            
            for trial in range(num_trials):
                if m == "full_kv":
                    cache = FullKVCache()
                elif m == "sliding_window":
                    cache = SlidingWindowCache(total_budget=B)
                elif m == "streaming_llm":
                    cache = StreamingLLMCache(total_budget=B, k_sink=4)
                elif m == "h2o":
                    cache = H2OHeavyHitterCache(total_budget=B, k_sink=4, w_local=B // 4)
                elif m == "synapse":
                    cache = SynapseTieredCache(total_budget=B, k_sink=4, w_local=B // 4, k_anchor=B // 2)
                    
                # Stream full conversation
                for t in range(N):
                    k = np.random.randn(8, 64).astype(np.float32)
                    v = np.random.randn(8, 64).astype(np.float32)
                    cache.insert_token(t, k, v, step=t)
                    
                    # If this is a Topic A fact, promote it in Synapse / update H2O
                    if t in topic_a_fact_steps:
                        if m == "synapse":
                            cache.promote_to_anchor(t, salience_score=80.0)
                        elif m == "h2o":
                            cache.update_step({t: [0.15] * 8})
                            
                    # While in Topic B and C, occasional back-reference to Topic A
                    if t > n_a and t % 500 == 0:
                        ref_fact = topic_a_fact_steps[trial % len(topic_a_fact_steps)]
                        if m == "synapse":
                            cache.promote_to_anchor(ref_fact, salience_score=75.0)
                        elif m == "h2o":
                            cache.update_step({ref_fact: [0.10] * 8})
                            
                # Check retention of all 5 Topic A facts at t = N
                for fact_idx in topic_a_fact_steps:
                    if cache.contains(fact_idx):
                        recovered_facts += 1
                        
            recovery_rate = (recovered_facts / total_facts) * 100.0
            row["topic_a_recovery_rate"][m] = {
                "recovery_pct": round(recovery_rate, 2),
                "recovered_facts": recovered_facts,
                "total_facts": total_facts
            }
            print(f"  --> {m:<15}: {recovery_rate:5.1f}% Topic A fact retention")
            
        results["topic_shift_data"].append(row)
        
    out_dir = r"c:\Users\rswar\OneDrive\Desktop\SynapseCache\data\benchmark_results"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "topic_shift_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 70)
    print(f"Topic Shift Results saved to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_topic_shift_benchmark()
