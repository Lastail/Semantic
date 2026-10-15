from pathlib import Path
import os,numpy as np,torch
from transformers import AutoTokenizer,AutoModel
from .base import BaseRepresentation
from ..metrics.neighbors import euclidean_two_nearest

class ContextualRepresentation(BaseRepresentation):
    """Transformer hidden states -> position mean -> Euclidean neighbours."""
    def __init__(self,model_path,device="cuda:0",dtype="float16",layer=-1,batch_size=4):
        self.device=device; self.layer=int(layer); self.batch_size=int(batch_size)
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token=self.tokenizer.eos_token
        dt=getattr(torch,dtype)
        self.model=AutoModel.from_pretrained(model_path,torch_dtype=dt,trust_remote_code=True).to(device).eval()

    def _vectors(self,texts,length):
        eligible=[]
        for text in texts:
            ids=self.tokenizer(text,add_special_tokens=True,truncation=False)["input_ids"]
            if len(ids)>=length: eligible.append(ids[:length])
        chunks=[]
        for s in range(0,len(eligible),self.batch_size):
            ids=torch.tensor(eligible[s:s+self.batch_size],dtype=torch.long,device=self.device)
            mask=torch.ones_like(ids)
            with torch.inference_mode():
                o=self.model(input_ids=ids,attention_mask=mask,use_cache=False,output_hidden_states=True,return_dict=True)
                x=o.hidden_states[self.layer].float().mean(dim=1).cpu().numpy().astype(np.float32,copy=False)
            if not np.isfinite(x).all(): raise RuntimeError("contextual vectors contain NaN/Inf")
            chunks.append(x)
        d=int(self.model.config.hidden_size)
        return np.concatenate(chunks) if chunks else np.empty((0,d),np.float32)

    def compute_neighbors(self,texts,length,cache_path=None):
        p=Path(cache_path) if cache_path else None
        if p and p.exists():
            with np.load(p,allow_pickle=False) as d:
                if "r1" in d and "r2" in d: return d["r1"],d["r2"]
        v=self._vectors(texts,length)
        if len(v)<3: raise ValueError(f"Only {len(v)} eligible texts for L={length}")
        r1,r2=euclidean_two_nearest(v)
        if p:
            p.parent.mkdir(parents=True,exist_ok=True); tmp=Path(str(p)+".tmp.npz")
            np.savez_compressed(tmp,r1=r1,r2=r2,n=len(v),length=length); os.replace(tmp,p)
        return r1,r2
