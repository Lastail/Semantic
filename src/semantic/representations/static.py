import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from .base import RepresentationExtractor

class StaticEmbeddingExtractor(RepresentationExtractor):
    """保持旧 units.py 的科学定义：input token embedding -> 前 L token -> mean pooling。"""
    def __init__(self, model_path, device="cuda:0", dtype="float16"):
        self.device=device; self.tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        dt=getattr(torch,dtype); self.model=AutoModel.from_pretrained(model_path,torch_dtype=dt,trust_remote_code=True).to(device).eval()
    def encode(self,texts,max_tokens):
        out=[]; emb_layer=self.model.get_input_embeddings()
        for text in texts:
            ids=self.tokenizer(text,return_tensors="pt")["input_ids"].to(self.device)[:,:max_tokens]
            with torch.inference_mode(): x=emb_layer(ids).float()[0].mean(dim=0)
            out.append(x.cpu().numpy())
        return np.asarray(out,dtype=np.float32)
