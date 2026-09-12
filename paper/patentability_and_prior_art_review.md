# Patentability, Prior-Art Review & Claims Draft: SynapseCache (KV-Optic)

**Author:** Sahil Rajesh Warke  
**Filing Recommendation:** Defensive Publication / Provisional Patent Specification  
**Subject Matter:** AI Systems / High-Efficiency Inference Acceleration / KV-Cache Memory Management

---

## 1. Prior-Art Comparative Matrix

| System | Prior-Art Ref | Mechanism | Core Limitation | SynapseCache Novelty |
| :--- | :--- | :--- | :--- | :--- |
| **StreamingLLM** | Xiao et al., ICLR 2024 | Initial attention sinks + FIFO local sliding window | Discards 100% of intermediate tokens; fails long-range factual recall (10-30% on randomized needles) | Introduces dynamic Relational Anchor Tier with cross-head consensus |
| **H2O** | Zhang et al., NeurIPS 2023 | Greedy accumulated attention mass across past steps | Susceptible to topic-drift staleness; no cross-head entropy normalization | Normalized Shannon entropy $p_{h,j}$ decouples diffuse relational hubs from transient spikes |
| **SnapKV** | Li et al., 2024 | Observation window attention clustering | Single observation window; static anchor selection | Multi-tier stateful feedback with continuous capacity rebalancing |
| **vLLM PagedAttention**| Kwon et al., SOSP 2023 | Virtual memory paging for KV cache fragmentation | Manages physical memory allocation, but does not evict tokens algorithmically | Algorithmic token pruning reducing logical cache size by 80-90% |

---

## 2. Patent Claims Draft

### Independent Claim 1 (Computer-Implemented Method)
1. A computer-implemented method for bounded-memory Key-Value (KV) cache management during autoregressive inference in a Multi-Head Attention transformer model, comprising:
   - receiving an incoming sequence of tokens and computing, for each token $j$ at generation step $t$, query-key attention weights across a plurality of query attention heads $h \in [1, H_Q]$;
   - computing a normalized cross-head attention probability distribution $p_{\cdot, j}$ across the plurality of query attention heads for token $j$;
   - calculating a Shannon cross-head attention entropy metric $\mathcal{H}_j$ from said normalized distribution;
   - evaluating a relational-anchor criterion based on a count of query attention heads exceeding a defined head attention threshold;
   - maintaining a tiered memory cache enforcing a strict capacity invariant $|\mathcal{C}_t| \le B$ across all generation steps $t \in [1, N]$, wherein said budget $B$ is constant and independent of total sequence length $N$; and
   - partitioning said tiered cache into:
     (i) a fixed invariant sink tier immune from eviction;
     (ii) a relational anchor tier holding tokens satisfying said relational-anchor criterion;
     (iii) a sliding local window tier holding a configurable number of most recent tokens; and
     (iv) an eviction candidate buffer holding tokens evicted from the local window tier, wherein candidates are pruned according to a composite salience metric derived from said Shannon entropy metric and attention mass.

### Independent Claim 2 (Computing System)
2. An inference acceleration computing system comprising:
   - a hardware accelerator comprising high-bandwidth memory (HBM);
   - one or more processors communicatively coupled to the hardware accelerator; and
   - a memory storing instructions that, when executed by the one or more processors, cause the system to:
     - project token embeddings into Query, Key, and Value representations across $H_Q$ query heads and $H_{\text{KV}}$ key-value heads, wherein $H_{\text{KV}} \le H_Q$;
     - store active Key and Value vectors in a bounded KV cache maintaining at most $B$ entries across all sequence steps;
     - detect relational anchors based on multi-head attention agreement and lock said anchors against recency-based eviction; and
     - perform scaled dot-product attention against said bounded KV cache, achieving at least an 80% reduction in HBM footprint while preserving attention accuracy.

### Independent Claim 3 (Non-Transitory Storage Medium)
3. A non-transitory computer-readable storage medium storing instructions that, when executed by a processor, perform the steps of Claim 1.

---

## 3. Defensive Publication vs. Patent Strategy

* **Recommendation:** Deposit this specification to **arXiv / Zenodo / IP.com** as a **Defensive Publication** immediately upon repository release.
* **Strategic Rationale:**
  1. Establishes prior art on the specific combination of *normalized cross-head Shannon entropy* and *tiered capacity invariants for KV compaction*, preventing competing corporate entities from patenting this exact method.
  2. Demonstrates research independence and systems authority for academic and research lab admissions.
