import torch

def masked_mean(hidden, attention_mask=None):
    if attention_mask is None: return hidden.mean(dim=1)
    mask=attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1)

def pool(hidden, attention_mask=None, method="mean"):
    if method=="mean": return masked_mean(hidden, attention_mask)
    if method=="last":
        if attention_mask is None: return hidden[:,-1]
        idx=attention_mask.sum(dim=1)-1
        return hidden[torch.arange(hidden.size(0),device=hidden.device),idx]
    raise ValueError(f"unknown pooling: {method}")
