# SynapseCache: Bounded-Memory Attention-Aware KV-Cache Compaction with Cross-Head Consensus

**Author:** Sahil Rajesh Warke  
**Affiliation:** Department of Computer Engineering, MET's Institute of Engineering, SPPU, Pune  
**Repository:** `https://github.com/rswarke1972-art/SynapseCache`  
**Live Interactive Telemetry:** `https://rswarke1972-art.github.io/SynapseCache/`  
**Artifact Classification:** IEEE Systems Architecture & Algorithmic Design Manuscript

---

## Abstract
Autoregressive Transformer inference in long-context regimes ($\ge 64,000$ tokens) is fundamentally bottlenecked by Key-Value (KV) cache memory growth ($O(N \cdot L \cdot H_{\text{KV}} \cdot d)$), exhausting GPU High-Bandwidth Memory (HBM) and bounding concurrency. Existing eviction heuristics either enforce recency windows that discard foundational long-range facts (StreamingLLM, Sliding Window) or rely on uncalibrated historical attention accumulation (H2O) that suffers from topic-drift staleness. In this paper, we present **SynapseCache (KV-Optic)**, a bounded-memory tiered KV-cache management architecture that guarantees a strict capacity invariant $|\mathcal{C}_t| \le B$ across all generation steps $t \in [1, N]$ independent of sequence length $N$. SynapseCache maintains online, bounded cross-head attention-mass statistics combined with normalized Shannon entropy and relational consensus, immune-locking multi-head relational anchors while evicting diffuse transient activations. In empirical evaluations against an unconstrained Full KV reference ceiling, SynapseCache achieves an **80.0% analytical KV-state reduction at a fixed 20% budget** (4.00 GB decimal / 3.73 GiB vs 20.00 GB decimal / 18.63 GiB at $N = 64,000$) with a **6.11x lower measured prototype attention-step latency** (36.17 ms vs 220.87 ms in local prototype benchmark profiling), while maintaining **100.0% retrieval recall** on evaluated randomized synthetic needle benchmarks ($p_{\text{needle}} \sim \mathcal{U}(0.05, 0.95)$) and **100.0% fact retention** across multi-turn topic migrations where pure recency and sink baselines suffer catastrophic forgetting (0.0% retention). Stress auditing exposes the system's operational boundary: unheralded tokens receiving zero observable early attention naturally slide out of the local window and are evicted, demonstrating that retention is an empirical consequence of observed multi-head consensus rather than an artificial oracle.

**Index Terms:** Transformers, Key-Value Cache, Long-Context Inference, Attention Entropy, Online Compaction, GPU Memory Optimization.

---

## I. Introduction
The scaling of transformer context windows to hundreds of thousands of tokens has unlocked document-level reasoning, code-base synthesis, and persistent agentic memory. However, this algorithmic expansion has collided with a severe hardware bottleneck: the **GPU Memory Wall**. 

During autoregressive generation, keys and values for all preceding tokens must be cached to prevent redundant quadratic forward passes. For modern architectures utilizing Grouped-Query Attention (GQA), the memory consumed by the KV cache scales as:
$$M_{\text{KV}} = 2 \cdot N \cdot L \cdot H_{\text{KV}} \cdot d_{\text{head}} \cdot b$$
where $b$ is element precision in bytes, $L$ is layer depth, and $H_{\text{KV}}$ is the number of key-value projection heads. For an 80-layer 70B-style architecture ($H_{\text{KV}} = 8, d_{\text{head}} = 128$) operating at $N = 64,000$ tokens in FP16, a single sequence requires **20.00 GB decimal (18.63 GiB)** of memory solely for cached activations. At $N = 128,000$, this exceeds **40 GB decimal**, severely limiting batch concurrency.

Prior eviction strategies exhibit fundamental limitations:
1. *Pure Recency (Sliding Window):* Evicts all tokens outside the most recent $w_{\text{local}}$ window, destroying early prompt constraints.
2. *Attention Sink Preservers (StreamingLLM):* Retains initial delimiter tokens ($k_{\text{sink}}$) and a local window, stabilizing perplexity but discarding intermediate factual content outside the retained region.
3. *Greedy Heavy Hitters (H2O):* Retains tokens based on uncalibrated historical attention accumulation, vulnerable to distractor overload during extended topic shifts.

To resolve this trade-off, we propose **SynapseCache**, an architectural framework that enforces strict memory bounds by construction while dynamically detecting and preserving cross-head relational anchors.

---

## II. System Architecture & Theoretical Invariants

### A. The Strict Capacity Invariant
Unlike heuristic memory managers that allow dynamic memory ballooning, SynapseCache establishes a deterministic upper bound enforced at every discrete generation step:
$$\boxed{|\mathcal{C}_t| \le B \quad \forall t \in [1, N]}$$
where the total cache budget $B$ is partitioned into four distinct operational tiers:
$$B = k_{\text{sink}} + k_{\text{anchor}} + w_{\text{local}} + k_{\text{candidate}}$$

### B. Tiered Cache Topology
1. **Invariant Sink Tier ($\mathcal{S}_{\text{sink}}$):** The initial $k_{\text{sink}}$ tokens ($t < k_{\text{sink}}$) receive 0% eviction probability, preserving initial system prompts and structural delimiters.
2. **Relational Anchor Tier ($\mathcal{S}_{\text{anchor}}$):** Holds tokens that satisfy the formal cross-head consensus criterion. Capped at capacity $k_{\text{anchor}}$.
3. **Sliding Local Window ($\mathcal{S}_{\text{local}}$):** Retains the $w_{\text{local}}$ most recent tokens in FIFO order to preserve local syntax and immediate conversational coherence.
4. **Eviction Candidate Buffer ($\mathcal{S}_{\text{candidate}}$):** Tokens pushed out of the local window that fail the anchor criterion, ranked dynamically by time-decayed attention salience.

### C. Normalized Cross-Head Attention Entropy
For token $j$ at step $t$, its raw attention weights across query heads $h \in [1, H_Q]$ are normalized into a probability distribution:
$$p_{h,j} = \frac{A_h(q_t, j)}{\sum_{h'=1}^{H_Q} A_{h'}(q_t, j) + \epsilon}$$
The cross-head Shannon entropy $\mathcal{H}_j$ is computed as:
$$\mathcal{H}_j = -\sum_{h=1}^{H_Q} p_{h,j} \log_2 (p_{h,j} + \epsilon) \bigg/ \log_2(H_Q)$$
Normalized by $\log_2(H_Q)$, $\mathcal{H}_j \in [0.0, 1.0]$. A high entropy indicates that a token acts as a diffuse relational hub across multiple representation subspaces, whereas low entropy indicates single-head specialization.

### D. Relational Anchor Criterion
A token is promoted to the immune anchor tier if it demonstrates cross-head agreement exceeding threshold $\tau_{\text{head}}$ across at least $\kappa$ distinct heads:
$$\text{IsAnchor}(j) = \mathbb{I}\left( \sum_{h=1}^{H_Q} \mathbb{I}(A_h(q_t, j) > \tau_{\text{head}}) \ge \kappa \right)$$

### E. Mathematical Clarification: Bounded Statistics vs Hash-Based Sketches
While probabilistic data stream systems (such as Count-Min Sketches or HyperLogLog in AegisStream) operate via randomized hash functions, SynapseCache maintains **online, bounded attention-mass statistics with cross-head entropy and exponential aging**. The term *sketch* in this context denotes a compact, lossy running summary of the full multi-head attention matrix that bounds state space to $O(B)$ rather than $O(N)$.

---

## III. Categorization of Scientific Evidence

To maintain rigorous scientific standards, we delineate the current state of verification across four explicit categories:

### 1. Proven by Implementation & Formal Testing
- **Strict Capacity Invariant:** $|\mathcal{C}_t| \le B \quad \forall t \in [1, N]$. Verified at every step across 5,000 continuous insertions and 7 adversarial stress conditions (14/14 automated tests passing).
- **Deterministic Sink Immunity:** $P(\text{eviction } | \text{ token } < k_{\text{sink}}) = 0.0$.
- **Graceful Under-Budget Adaptation:** Correct operation when $B \le k_{\text{sink}} + w_{\text{local}}$.

### 2. Empirically Demonstrated on Evaluated Workloads
- **Analytical KV-State Reduction:** 80.0% state reduction at a fixed 20% budget ($4.00\text{ GB decimal} / 3.73\text{ GiB}$ vs $20.00\text{ GB decimal} / 18.63\text{ GiB}$ at $N=64,000$).
- **Simulated Step Latency:** 6.11x lower prototype attention-step latency in local benchmark profiling (36.17 ms vs 220.87 ms at $N=64,000$).
- **Randomized Needle Recall:** 100% retrieval on the current synthetic benchmark with uniform needle placements $p_{\text{needle}} \sim \mathcal{U}(0.05, 0.95)$ across 10 random seeds.
- **Topic-Shift Fact Retention:** 100% retention on the evaluated multi-turn topic migration workload across 12,000 tokens (where pure recency drops to 0.0%).
- **Multi-Needle Retention Recall:** 100% retention recall for $K \le 64$ under $B = 128$.

### 3. Empirically Demonstrated Operational Limitations
- **Cold-Needle Vulnerability:** Facts receiving zero observable attention at insertion are treated as ordinary local tokens, sliding out of $w_{\text{local}}$ and facing eviction (evicted at step 324 of 2,000 in our adversarial audit).
- **Retention Precision Trade-off:** Under fixed budgets ($B=128$), retention precision scales roughly as $P_R \approx K / 128$. At $K=80$, $R_R = 100\%$ but $P_R = 64.5\%$, leaving 35.5% of the cache occupied by non-needle background tokens.

### 4. Not Yet Established (Future Work)
- End-to-end inference latency on physical 70B GPU clusters with compiled FlashAttention/Triton kernels.
- Universal empirical retrieval guarantee ($\ge 98.5\%$) across arbitrary real-world LLMs and unconstrained natural language corpuses.
- Formal analytical $\epsilon$-approximation error bounds on downstream logits.
- Formal patent grant.

---

## IV. Experimental Results & Stress Audits

### A. Randomized Needle-in-a-Haystack Retrieval Sweep
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

### B. Analytical KV-Memory Model for a 70B-Style Architecture ($L=80, H_{\text{KV}}=8, d_{\text{head}}=128$, FP16)
> *Clarification:* These latency values reflect local prototype attention simulation on CPU/NumPy and are not measurements of end-to-end inference latency on a distributed 70B GPU serving cluster.

| Context ($N$) | Full KV Memory | SynapseCache (20% Budget) | State Reduction | Full KV Step Latency | SynapseCache Step Latency | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2,000** | 625.00 MB (0.58 GiB) | 125.00 MB (0.12 GiB) | 80.0% | 5.90 ms | 1.03 ms | **5.75x** |
| **4,000** | 1,250.00 MB (1.16 GiB) | 250.00 MB (0.23 GiB) | 80.0% | 11.26 ms | 1.96 ms | **5.75x** |
| **8,000** | 2,500.00 MB (2.33 GiB) | 500.00 MB (0.47 GiB) | 80.0% | 25.94 ms | 4.50 ms | **5.77x** |
| **16,000** | 5,000.00 MB (4.66 GiB) | 1,000.00 MB (0.93 GiB) | 80.0% | 45.46 ms | 9.15 ms | **4.97x** |
| **32,000** | 10,000.00 MB (9.31 GiB) | 2,000.00 MB (1.86 GiB) | 80.0% | 119.22 ms | 21.71 ms | **5.49x** |
| **64,000** | 20,000.00 MB (18.63 GiB) | 4,000.00 MB (3.73 GiB) | **80.0%** | 220.87 ms | 36.17 ms | **6.11x** |

### C. Multi-Needle Saturation & Retention Trade-offs ($B = 128, k_{\text{anchor}} = 64$)
| Needle Count ($K$) | Retained Needles | Retention Recall ($R_R$) | Retention Precision ($P_R$) | Total Cache Size |
| :--- | :--- | :--- | :--- | :--- |
| **2** | 2 / 2 | **100.0%** | 1.6% | 128 |
| **5** | 5 / 5 | **100.0%** | 4.0% | 128 |
| **10** | 10 / 10 | **100.0%** | 8.1% | 128 |
| **20** | 20 / 20 | **100.0%** | 16.1% | 128 |
| **40** | 40 / 40 | **100.0%** | 32.3% | 128 |
| **80** | 80 / 80 | **100.0%** | 64.5% | 128 |

---

## V. Conclusion
SynapseCache presents a mathematically bounded approach to KV-cache compaction. By coupling a strict capacity invariant $|\mathcal{C}_t| \le B$ with cross-head attention consensus, SynapseCache maintains long-range relational facts while reducing analytical memory footprints by 80.0%. Acknowledging the cold-needle failure boundary grounds the system in rigorous empirical reality, defining clear paths for future research in predictive pre-retention.
