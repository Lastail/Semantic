"""Mean of selected Transformer hidden-state layer across token positions."""
from __future__ import annotations
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from .mean_base import MeanVectorRepresentation

class ContextualMeanRepresentation(MeanVectorRepresentation):
    method = 'contextual_mean'
    def __init__(self, model_path, device='cuda:0', dtype='float16', layer=-1,
                 batch_size=4, normalize=False, **kwargs):
        super().__init__(model_path, device, dtype, batch_size, normalize)
        self.layer = int(layer)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_path, dtype=getattr(torch, dtype), trust_remote_code=True
        ).to(device).eval()
        self.vector_dim = int(self.model.config.hidden_size)
        k = int(self.model.config.num_hidden_layers)
        if not -(k+1) <= self.layer <= k:
            raise ValueError(f'layer={self.layer} outside hidden_states indices 0..{k} or -1..-{k+1}')

    def _encode_batch(self, rows):
        ids = torch.tensor(rows, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                             use_cache=False, output_hidden_states=True, return_dict=True)
            x = out.hidden_states[self.layer].float().mean(dim=1)
        return x.cpu().numpy().astype(np.float32)
