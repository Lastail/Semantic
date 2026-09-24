import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from .base import RepresentationExtractor
from .pooling import pool

class ContextualExtractor(RepresentationExtractor):
    """完整 Transformer forward；支持任意 hidden-state layer 与 mean/last pooling。"""
    def __init__(self,model_path,device="cuda:0",dtype="float16",layer=-1,pooling="mean",batch_size=4):
        self.device=device; self.layer=layer; self.pooling=pooling; self.batch_size=batch_size
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token=self.tokenizer.eos_token
        dt=getattr(torch,dtype); self.model=AutoModel.from_pretrained(model_path,torch_dtype=dt,trust_remote_code=True).to(device).eval()
    def encode(self,texts,max_tokens):
        eligible=[]
        for t in texts:
            ids=self.tokenizer(t,add_special_tokens=True,truncation=False)["input_ids"]
            if len(ids)>=max_tokens: eligible.append(ids[:max_tokens])
        chunks=[]
        for s in range(0,len(eligible),self.batch_size):
            ids=torch.tensor(eligible[s:s+self.batch_size],dtype=torch.long,device=self.device); mask=torch.ones_like(ids)
            with torch.inference_mode():
                o=self.model(input_ids=ids,attention_mask=mask,use_cache=False,output_hidden_states=True,return_dict=True)
                h=o.hidden_states[self.layer]
                x=pool(h,mask,self.pooling).float().cpu().numpy()
            if not np.isfinite(x).all(): raise RuntimeError("contextual vectors contain NaN/Inf")
            chunks.append(x)
        hidden=self.model.config.hidden_size
        return np.concatenate(chunks).astype(np.float32) if chunks else np.empty((0,hidden),np.float32)
