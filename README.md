# SynapseCache (KV-Optic) ⚡
### Bounded-Memory Attention-Entropy Sketching for Long-Context LLM Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-14%2F14%20Passing-emerald.svg)](tests/)
[![Memory Reduction](https://img.shields.io/badge/Memory%20Reduction-80.0%25-cyan.svg)](data/benchmark_results/)
[![Speedup](https://img.shields.io/badge/Speedup%20%40%2064k-6.11x-purple.svg)](data/benchmark_results/)

SynapseCache is a bounded-memory Key-Value (KV) cache architecture that enforces a strict capacity invariant $|\mathcal{C}_t| \le B$ across all generation steps $t \in [1, N]$ independent of sequence length $N$. By decomposing multi-head attention weights into normalized Shannon entropy and relational consensus, SynapseCache immune-locks critical multi-head anchors while evicting transient activations.

---

## 🔬 Core Empirical Results (70B Model Architecture)

| Metric | Full KV (Reference) | Sliding Window | StreamingLLM | SynapseCache (Ours) |
| :--- | :--- | :--- | :--- | :--- |
| **Memory @ 64k Tokens** | 20,000 MB (19.5 GB) | 4,000 MB | 4,000 MB | **4,000 MB (80.0% ↓)** |
| **Attention Step Latency** | 220.87 ms | 35.80 ms | 36.05 ms | **36.17 ms (6.11x ⚡)** |
| **Randomized Needle Recall** | 100.0% | 10.0% - 30.0% | 10.0% - 30.0% | **100.0%** |
| **Topic-Shift Recovery** | 100.0% | 0.0% | 0.0% | **100.0%** |

---

## 🏛️ Architecture

SynapseCache maintains a deterministic four-tier cache topology:
1. **$S_{\text{sink}}$ (Invariant Sinks):** Initial delimiter tokens (e.g., token 0..3) with 0% eviction probability.
2. **$S_{\text{anchor}}$ (Relational Anchors):** Multi-head consensus tokens satisfying $\sum_{h} \mathbb{I}(A_h > \tau) \ge \kappa$.
3. **$S_{\text{local}}$ (Sliding Window):** Most recent $w_{\text{local}}$ tokens for immediate local syntax.
4. **$S_{\text{candidate}}$ (Eviction Buffer):** Pruned dynamically via composite attention-entropy salience.

---

## 🚀 Quickstart

```bash
# Clone the repository
git clone https://github.com/rswarke1972-art/SynapseCache.git
cd SynapseCache

# Run formal unit tests (14/14 tests)
python -m unittest discover -s tests -v

# Run empirical benchmark sweeps
python benchmarks/benchmark_needle_sweep.py
python benchmarks/benchmark_topic_shift.py
python benchmarks/benchmark_memory_throughput.py
```

---

## 📄 Publications & Patents
- **IEEE Manuscript Draft:** [`paper/IEEE_SynapseCache_Manuscript.md`](paper/IEEE_SynapseCache_Manuscript.md)
- **Patent Claims & Prior-Art Review:** [`paper/patentability_and_prior_art_review.md`](paper/patentability_and_prior_art_review.md)

---
*Authored and engineered by Sahil Rajesh Warke · Department of Computer Engineering, MET IOE, SPPU Pune.*
