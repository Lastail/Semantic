"""Static mean using ONLY the input embedding weight; no full model load."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from .mean_base import MeanVectorRepresentation


class StaticMeanRepresentation(MeanVectorRepresentation):
    method = "static_mean"

    def __init__(self, model_path, device="cuda:0", dtype="float16",
                 batch_size=4, normalize=False, **kwargs):
        super().__init__(model_path, device, dtype, batch_size, normalize)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.embedding = self._load_embedding_only()
        self.vector_dim = int(self.embedding.embedding_dim)
        print(f"[static_mean] loaded embedding only: {tuple(self.embedding.weight.shape)}",
              flush=True)

    def _load_embedding_only(self):
        folder = Path(self.model_path)
        candidates = (
            "model.embed_tokens.weight",
            "model.model.embed_tokens.weight",
            "transformer.wte.weight",
            "embed_tokens.weight",
        )
        index_path = folder / "model.safetensors.index.json"
        shard = None
        name = None

        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
            for candidate in candidates:
                if candidate in weight_map:
                    name, shard = candidate, folder / weight_map[candidate]
                    break
        else:
            shards = sorted(folder.glob("*.safetensors"))
            for candidate in candidates:
                for file in shards:
                    with safe_open(str(file), framework="pt", device="cpu") as f:
                        if candidate in f.keys():
                            name, shard = candidate, file
                            break
                if shard is not None:
                    break

        if shard is None:
            raise RuntimeError(
                "Could not locate input embedding tensor in safetensors weights. "
                "This loader supports Qwen-style safetensors checkpoints; "
                "verify the tensor name and model format."
            )

        with safe_open(str(shard), framework="pt", device="cpu") as f:
            weight = f.get_tensor(name)

        if weight.ndim != 2:
            raise ValueError(f"Embedding tensor has shape {tuple(weight.shape)}")

        # Keep only the embedding weights on the requested device.
        torch_dtype = getattr(torch, self.dtype)
        module = torch.nn.Embedding.from_pretrained(
            weight.to(dtype=torch_dtype), freeze=True
        ).to(self.device).eval()
        return module

    def _encode_batch(self, rows):
        ids = torch.tensor(rows, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            x = self.embedding(ids).float().mean(dim=1)
        return x.cpu().numpy().astype(np.float32, copy=False)
