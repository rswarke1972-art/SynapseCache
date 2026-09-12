"""Attention Entropy and Salience Extraction Module for SynapseCache.

Formulates mathematically sound cross-head probability distributions,
Shannon entropy dispersion, mean attention mass, and configurable relational-anchor criteria.
"""

import math
from typing import Dict, List, Optional, Tuple, NamedTuple


class SalienceScores(NamedTuple):
    """Container for decomposed token saliency features."""
    mean_mass: float
    entropy: float
    is_anchor: bool
    composite_salience: float


def compute_cross_head_distribution(
    head_attentions: List[float],
    epsilon: float = 1e-12
) -> List[float]:
    """Normalizes attention weights across query heads for a specific key token j.
    
    p_{h, j} = A_h(q_t, j) / (sum_{h'=1}^{H_Q} A_{h'}(q_t, j) + epsilon)
    
    Args:
        head_attentions: Raw attention weights for token j from each head h in [1, H_Q].
        epsilon: Numerical stability constant.
        
    Returns:
        List of normalized probabilities summing to 1.0.
    """
    total_mass = sum(head_attentions)
    denominator = total_mass + epsilon
    return [max(0.0, float(a)) / denominator for a in head_attentions]


def compute_cross_head_entropy(
    head_attentions: List[float],
    epsilon: float = 1e-12
) -> float:
    """Computes Shannon entropy across heads for token j:
    
    H_j = - sum_{h=1}^{H_Q} p_{h, j} * log_2(p_{h, j} + epsilon)
    
    High entropy indicates diffuse attention across multiple heads (potential relational hub).
    Low entropy indicates focused, single-head specialized attention.
    """
    if not head_attentions:
        return 0.0
        
    p_dist = compute_cross_head_distribution(head_attentions, epsilon)
    entropy = 0.0
    for p in p_dist:
        if p > 0.0:
            entropy -= p * math.log2(p + epsilon)
            
    # Normalize by log2(H_Q) so entropy is bounded in [0.0, 1.0] if H_Q > 1
    num_heads = len(head_attentions)
    if num_heads > 1:
        max_entropy = math.log2(num_heads)
        if max_entropy > 0.0:
            entropy = max(0.0, min(1.0, entropy / max_entropy))
    else:
        entropy = 0.0
        
    return entropy


def compute_mean_attention_mass(head_attentions: List[float]) -> float:
    """Computes average attention mass across all query heads for token j:
    
    M_j = (1 / H_Q) * sum_{h=1}^{H_Q} A_h(q_t, j)
    """
    if not head_attentions:
        return 0.0
    return sum(head_attentions) / len(head_attentions)


def evaluate_anchor_criterion(
    head_attentions: List[float],
    tau_head: float = 0.05,
    kappa_heads: int = 2
) -> bool:
    """Evaluates whether token j satisfies the configurable relational-anchor criterion:
    
    IsAnchor(j) = [ sum_{h=1}^{H_Q} I(A_h(q_t, j) > tau_head) >= kappa ]
    
    A token is an anchor if at least kappa distinct heads allocate attention > tau_head.
    """
    qualifying_heads = sum(1 for a in head_attentions if a > tau_head)
    return qualifying_heads >= kappa_heads


def compute_token_salience(
    head_attentions: List[float],
    gamma: float = 0.6,
    tau_head: float = 0.05,
    kappa_heads: int = 2,
    epsilon: float = 1e-12
) -> SalienceScores:
    """Calculates full decomposed saliency metric for a single key token.
    
    Composite Salience = gamma * M_j + (1 - gamma) * H_j
    
    Treated as an experimental scoring function to evaluate empirical utility.
    """
    mean_m = compute_mean_attention_mass(head_attentions)
    entropy = compute_cross_head_entropy(head_attentions, epsilon)
    is_anchor = evaluate_anchor_criterion(head_attentions, tau_head, kappa_heads)
    composite = (gamma * mean_m) + ((1.0 - gamma) * entropy)
    
    return SalienceScores(
        mean_mass=mean_m,
        entropy=entropy,
        is_anchor=is_anchor,
        composite_salience=composite
    )


class AttentionScorer:
    """Stateful attention feature extraction engine with exponential historical aging."""
    
    def __init__(
        self,
        gamma: float = 0.6,
        decay_lambda: float = 0.95,
        tau_head: float = 0.05,
        kappa_heads: int = 2,
        epsilon: float = 1e-12
    ):
        self.gamma = gamma
        self.decay_lambda = decay_lambda
        self.tau_head = tau_head
        self.kappa_heads = kappa_heads
        self.epsilon = epsilon
        
        # Token index -> accumulated historical salience
        self.accumulated_salience: Dict[int, float] = {}
        # Token index -> cumulative anchor qualification count
        self.anchor_hit_counts: Dict[int, int] = {}
        
    def score_step(
        self,
        step_attentions: Dict[int, List[float]]
    ) -> Dict[int, SalienceScores]:
        """Scores all active key tokens for the current generation step.
        
        Args:
            step_attentions: Dict mapping token_index -> list of attention weights across heads.
            
        Returns:
            Dict mapping token_index -> SalienceScores namedtuple.
        """
        results: Dict[int, SalienceScores] = {}
        
        # Age existing accumulated salience
        for tid in list(self.accumulated_salience.keys()):
            self.accumulated_salience[tid] *= self.decay_lambda
            
        for tid, head_weights in step_attentions.items():
            scores = compute_token_salience(
                head_weights,
                gamma=self.gamma,
                tau_head=self.tau_head,
                kappa_heads=self.kappa_heads,
                epsilon=self.epsilon
            )
            
            curr_accum = self.accumulated_salience.get(tid, 0.0)
            self.accumulated_salience[tid] = curr_accum + scores.composite_salience
            
            if scores.is_anchor:
                self.anchor_hit_counts[tid] = self.anchor_hit_counts.get(tid, 0) + 1
                
            results[tid] = scores
            
        return results
        
    def get_accumulated_score(self, token_id: int) -> float:
        """Returns the time-decayed accumulated salience for a token."""
        return self.accumulated_salience.get(token_id, 0.0)
        
    def prune_token(self, token_id: int) -> None:
        """Cleans state when a token is evicted from the cache."""
        self.accumulated_salience.pop(token_id, None)
        self.anchor_hit_counts.pop(token_id, None)
