import torch
import torch.nn.functional as F


def top_k_sample(logits: torch.Tensor, k: int, t: float = 1.0) -> torch.Tensor:
    """Sample from the top k logits with temperature t"""
    assert k > 0, "k must be greater than 0"
    assert t > 0, "t must be greater than 0"
    logits = logits / t
    top_k_logits, top_k_indices = torch.topk(logits, k)
    top_k_probs = F.softmax(top_k_logits, dim=-1)
    sampled_index = torch.multinomial(top_k_probs, 1)
    sampled_token = top_k_indices.gather(0, sampled_index)
    return sampled_token
