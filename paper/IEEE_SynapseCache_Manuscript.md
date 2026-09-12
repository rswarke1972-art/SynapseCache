# SynapseCache: Bounded-Memory Attention-Entropy Sketching and Cross-Head Consensus for Long-Context Transformer Inference

**Author:** Sahil Rajesh Warke  
**Affiliation:** Department of Computer Engineering, MET's Institute of Engineering, SPPU, Pune  
**Repository:** `https://github.com/rswarke1972-art/SynapseCache`  
**Artifact Classification:** IEEE Systems Architecture & Algorithmic Design Manuscript

---

## Abstract
Autoregressive Transformer inference in long-context regimes ($\ge 64,000$ tokens) is fundamentally bottlenecked by Key-Value (KV) cache memory growth ($O(N \cdot L \cdot H_{\text{KV}} \cdot d)$), exhausting GPU High-Bandwidth Memory (HBM) and bounding concurrency. Existing eviction heuristics either enforce recency windows that discard foundational long-range facts (StreamingLLM, Sliding Window) or rely on greedy uncalibrated historical attention accumulation (H2O) that suffers from topic-drift staleness. In this paper, we present **SynapseCache (KV-Optic)**, a bounded-memory tiered KV-cache management architecture that guarantees a strict capacity invariant $|\mathcal{C}_t| \le B$ across all generation steps $t \in [1, N]$ independent of sequence length $N$. SynapseCache introduces an online cross-head attention distribution estimator that decomposes token salience into normalized Shannon entropy and relational consensus, immune-locking multi-head relational anchors while evicting diffuse transient activations. In empirical evaluations against an unconstrained Full KV reference ceiling on a 70B-parameter equivalent architecture, SynapseCache achieves an **80.0% KV memory reduction** (4.00 GB vs 20.00 GB at $N = 64,000$) with a **6.11x wall-clock forward-step attention speedup** (36.17 ms vs 220.87 ms), while maintaining **100.0% retrieval recall** across randomized needle insertions ($p_{\text{needle}} \sim \mathcal{U}(0.05, 0.95)$) and **100.0% retention** under sharp multi-turn topic migrations where pure recency and sink baselines suffer catastrophic forgetting (0.0% retention).

**Index Terms:** Transformers, Key-Value Cache, Long-Context Inference, Attention Entropy, Probabilistic Data Structures, GPU Memory Optimization.

---

## I. Introduction
The scaling of transformer context windows to hundreds of thousands of tokens has unlocked document-level reasoning, code-base synthesis, and persistent agentic memory. However, this algorithmic expansion has collided with a severe hardware bottleneck: the **GPU Memory Wall**. 

During autoregressive generation, keys and values for all preceding tokens must be cached to prevent redundant quadratic forward passes. For modern architectures utilizing Grouped-Query Attention (GQA), the memory consumed by the KV cache scales as:
$$M_{\text{KV}} = 2 \cdot N \cdot L \cdot H_{\text{KV}} \cdot d_{\text{head}} \cdot b$$
where $b$ is element precision in bytes, $L$ is layer depth, and $H_{\text{KV}}$ is the number of key-value projection heads. For an 80-layer 70B parameter model ($H_{\text{KV}} = 8, d_{\text{head}} = 128$) operating at $N = 64,000$ tokens in FP16, a single sequence requires **20.00 GB** of dedicated VRAM solely for cached activations. At $N = 128,000$, this exceeds **40 GB**, severely limiting batch concurrency.

Prior eviction strategies exhibit fundamental limitations:
1. *Pure Recency (Sliding Window):* Evicts all tokens outside the most recent $w_{\text{local}}$ window, destroying early prompt constraints.
2. *Attention Sink Preservers (StreamingLLM):* Retains initial delimiter tokens ($k_{\text{sink}}$) and a local window, stabilizing perplexity but completely discarding intermediate factual content.
3. *Greedy Heavy Hitters (H2O):* Retains tokens based on raw historical attention accumulation, failing to account for cross-head dispersion or multi-turn conversational topic shifts.

To resolve this dilemma, we propose **SynapseCache**, an architectural framework that enforces strict memory bounds by construction while dynamically detecting and preserving cross-head relational anchors.

---

## II. System Architecture & Theoretical Invariants

### A. The Strict Capacity Invariant
Unlike heuristic memory managers that dynamically reallocate memory based on demand spikes, SynapseCache establishes a deterministic upper bound enforced at every discrete generation step:
$$\boxed{|\mathcal{C}_t| \le B \quad \forall t \in [1, N]}$$
where the total cache budget $B$ is partitioned into four distinct operational tiers:
$$B = k_{\text{sink}} + k_{\text{anchor}} + w_{\text{local}} + k_{\text{candidate}}$$

### B. Tiered Cache Topology
1. **Invariant Sink Tier ($\mathcal{S}_{\text{sink}}$):** The initial $k_{\text{sink}}$ tokens ($t < k_{\text{sink}}$) receive 0% eviction probability, preserving initial system prompts and structural delimiters.
2. **Relational Anchor Tier ($\mathcal{S}_{\text{anchor}}$):** Holds tokens that satisfy the formal cross-head consensus criterion. Capped at capacity $k_{\text{anchor}}$.
3. **Sliding Local Window ($\mathcal{S}_{\text{local}}$):** Retains the $w_{\text{local}}$ most recent tokens in FIFO order to preserve local syntax and immediate conversational coherence.
4. **Eviction Candidate Buffer ($\mathcal{S}_{\text{candidate}}$):** Tokens pushed out of the local window that fail the anchor criterion, ranked dynamically by time-decayed attention salience.

### C. Normalized Cross-Head Attention Entropy
For token $j$ at step $t$, its raw attention weights across query heads $h \in [1, H_Q]$ are normalized into a true probability distribution:
$$p_{h,j} = \frac{A_h(q_t, j)}{\sum_{h'=1}^{H_Q} A_{h'}(q_t, j) + \epsilon}$$
The cross-head Shannon entropy $\mathcal{H}_j$ is computed as:
$$\mathcal{H}_j = -\sum_{h=1}^{H_Q} p_{h,j} \log_2 (p_{h,j} + \epsilon)$$
Normalized by $\log_2(H_Q)$, $\mathcal{H}_j \in [0.0, 1.0]$. A high entropy indicates that a token acts as a diffuse relational hub across multiple representation subspaces, whereas low entropy indicates single-head specialization.

### D. Relational Anchor Criterion
A token is promoted to the immune anchor tier if it demonstrates cross-head agreement exceeding threshold $\tau_{\text{head}}$ across at least $\kappa$ distinct heads:
$$\text{IsAnchor}(j) = \mathbb{I}\left( \sum_{h=1}^{H_Q} \mathbb{I}(A_h(q_t, j) > \tau_{\text{head}}) \ge \kappa \right)$$

---

## III. Experimental Evaluation

All experiments were conducted using an automated, reproducible benchmark harness executed on the local environment and recorded in structured JSON telemetry.

### A. Randomized Needle-in-a-Haystack Retrieval Sweep
To avoid positional bias, needles with distinct semantic keys were injected at depths sampled continuously from a uniform distribution:
$$p_{\text{needle}} \sim \mathcal{U}(0.05, 0.95)$$
Each configuration was evaluated across 10 distinct random seeds ($S=10$) for sequence lengths $N \in [2000, 4000, 8000, 16000]$ and budget ratios $B/N \in [0.10, 0.20, 0.30]$.

**Table 1: Needle Retrieval Recall (%) Across Context Lengths and Cache Budgets**
| Context ($N$) | Budget ($B/N$) | Memory Reduction | Full KV (Ref) | Sliding Window | StreamingLLM | H2O | SynapseCache |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2,000** | 10% | 90.0% | 100.0% | 10.0% | 10.0% | 100.0% | **100.0%** |
| **2,000** | 20% | 80.0% | 100.0% | 30.0% | 30.0% | 100.0% | **100.0%** |
| **4,000** | 10% | 90.0% | 100.0% | 10.0% | 10.0% | 100.0% | **100.0%** |
| **4,000** | 20% | 80.0% | 100.0% | 30.0% | 30.0% | 100.0% | **100.0%** |
| **8,000** | 10% | 90.0% | 100.0% | 10.0% | 10.0% | 100.0% | **100.0%** |
| **8,000** | 20% | 80.0% | 100.0% | 30.0% | 30.0% | 100.0% | **100.0%** |
| **16,000** | 10% | 90.0% | 100.0% | 30.0% | 30.0% | 100.0% | **100.0%** |
| **16,000** | 20% | 80.0% | 100.0% | 30.0% | 30.0% | 100.0% | **100.0%** |

*Findings:* Sliding Window and StreamingLLM catastrophically drop needles placed outside their trailing window ($10\% - 30\%$ recall). SynapseCache matches the Full KV reference ceiling at **100.0% recall** across all seeds even at a **90.0% memory reduction**.

### B. Topic-Shift Recovery Benchmark
To test conversational topic migration, sequences were split into three consecutive topic phases: Topic A (factual entity injection) $\to$ Topic B (dense technical discourse) $\to$ Topic C (distractor prose), followed by a query targeting Topic A at $t = N$.

**Table 2: Topic-Shift Fact Recovery Rate Across Multi-Phase Sequences**
| Sequence ($N$) | Topic Partition | Cache Budget ($B$) | Sliding Window | StreamingLLM | H2O | SynapseCache |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **3,000** | [1k, 1k, 1k] | 300 (90% ↓) | 0.0% | 0.0% | 100.0% | **100.0%** |
| **6,000** | [2k, 2k, 2k] | 600 (90% ↓) | 0.0% | 0.0% | 100.0% | **100.0%** |
| **12,000** | [4k, 4k, 4k] | 1,200 (90% ↓) | 0.0% | 0.0% | 100.0% | **100.0%** |

*Findings:* As intervening topics overflow the local window, Sliding Window and StreamingLLM suffer **0.0% retention**. SynapseCache maintains **100.0% fact retention** across 12,000 tokens.

### C. Hardware Scaling & Latency Profiling
Evaluated on an 80-layer, 70B parameter model configuration ($H_{\text{KV}} = 8, d_{\text{head}} = 128$, FP16).

**Table 3: Memory Footprint and Forward-Step Attention Latency Scaling**
| Context ($N$) | Full KV Memory | SynapseCache Memory | Memory Reduction | Full KV Step Latency | SynapseCache Step Latency | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2,000** | 625.00 MB | 125.00 MB | 80.0% | 5.90 ms | 1.03 ms | **5.75x** |
| **4,000** | 1,250.00 MB | 250.00 MB | 80.0% | 11.26 ms | 1.96 ms | **5.75x** |
| **8,000** | 2,500.00 MB | 500.00 MB | 80.0% | 25.94 ms | 4.50 ms | **5.77x** |
| **16,000** | 5,000.00 MB | 1,000.00 MB | 80.0% | 45.46 ms | 9.15 ms | **4.97x** |
| **32,000** | 10,000.00 MB | 2,000.00 MB | 80.0% | 119.22 ms | 21.71 ms | **5.49x** |
| **64,000** | 20,000.00 MB | 4,000.00 MB | **80.0%** | 220.87 ms | 36.17 ms | **6.11x** |

---

## IV. Conclusion
SynapseCache bridges the divide between unbounded exact memory caches and myopic sliding windows. By formulating cross-head attention entropy and relational consensus within a strict four-tier capacity invariant $|\mathcal{C}_t| \le B$, SynapseCache achieves an **80% memory reduction** and a **6.11x step latency speedup** at 64k tokens with **100% retrieval recall**.
