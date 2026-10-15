from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from .base import BaseRepresentation

class PositionalRepresentation(BaseRepresentation):
    """Input token embeddings only; flattened Euclidean metric.

    d(A,B) = sqrt(sum_{i,k} (E[token_A[i],k]-E[token_B[i],k])**2).
    No Transformer forward, pooling or normalization. Only r1/r2 are saved.
    """
    def __init__(self, model_path, device='cuda:0', dtype='float16',
                 text_block_size=8, position_block_size=128, **kwargs):
        self.model_path=str(Path(model_path).expanduser().resolve())
        self.device=torch.device(device)
        self.text_block_size=int(text_block_size)
        self.position_block_size=int(position_block_size)
        if min(self.text_block_size,self.position_block_size)<1:
            raise ValueError('Block sizes must be positive')
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_path,trust_remote_code=True)
        # Generic HF loading: weights of the full model are loaded, but forward is NEVER called.
        self.model=AutoModel.from_pretrained(self.model_path,dtype=getattr(torch,dtype),trust_remote_code=True).to(self.device).eval()
        self.embedding=self.model.get_input_embeddings()
        if self.embedding is None: raise RuntimeError('No input embedding layer')

    def _tokenize(self,texts,length):
        rows=[]
        for text in texts:
            ids=self.tokenizer(text,add_special_tokens=True,truncation=False)['input_ids']
            if len(ids)>=length: rows.append(ids[:length])
        if len(rows)<3: raise ValueError(f'Only {len(rows)} eligible texts for L={length}')
        return torch.tensor(rows,dtype=torch.long)

    def _identity(self,ids,length):
        h=hashlib.sha256()
        h.update(json.dumps({'model':self.model_path,'length':length,'metric':'embedding_flat_l2_v2'},sort_keys=True).encode())
        h.update(ids.numpy().astype('<i8',copy=False).tobytes())
        return h.hexdigest()

    @staticmethod
    def _merge(r1,r2,candidates):
        z=np.concatenate([np.stack((r1,r2),axis=1),candidates],axis=1)
        two=np.partition(z,1,axis=1)[:,:2]
        two.sort(axis=1)
        return two[:,0],two[:,1]

    def _distance_block(self,A,B,length):
        accum=torch.zeros((len(A),len(B)),device=self.device,dtype=torch.float64)
        with torch.inference_mode():
            for p0 in range(0,length,self.position_block_size):
                p1=min(length,p0+self.position_block_size)
                a=self.embedding(A[:,p0:p1].to(self.device)).to(torch.float64).flatten(1)
                b=self.embedding(B[:,p0:p1].to(self.device)).to(torch.float64).flatten(1)
                sq=(a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*(a@b.T)
                accum+=sq.clamp_min(0)
        return torch.sqrt(accum.clamp_min(0)).cpu().numpy()

    def compute_neighbors(self,texts,length,cache_path=None):
        length=int(length)
        ids=self._tokenize(texts,length)
        identity=self._identity(ids,length)
        p=Path(cache_path) if cache_path is not None else None
        if p is not None and p.exists():
            try:
                with np.load(p,allow_pickle=False) as d:
                    if str(d['identity'].item())==identity and len(d['r1'])==len(ids):
                        return d['r1'].copy(),d['r2'].copy()
            except (ValueError,KeyError,OSError): pass
        n=len(ids); B=self.text_block_size
        r1=np.full(n,np.inf); r2=np.full(n,np.inf)
        starts=list(range(0,n,B))
        # Small atomic checkpoint; no hidden-state file.
        cp=Path(str(p)+'.progress.npz') if p is not None else None
        first=0
        if cp is not None and cp.exists():
            try:
                with np.load(cp,allow_pickle=False) as d:
                    if str(d['identity'].item())==identity and int(d['block_size'])==B and len(d['r1'])==n:
                        first=int(d['next_outer'])
                        if not 0<=first<=len(starts): raise ValueError('Invalid checkpoint')
                        r1,r2=d['r1'].copy(),d['r2'].copy()
            except (ValueError,KeyError,OSError): first=0
        for ai in tqdm(range(first,len(starts)),desc=f'positional NN L={length}'):
            i0=starts[ai]; i1=min(i0+B,n)
            for j0 in starts[ai:]:
                j1=min(j0+B,n)
                dist=self._distance_block(ids[i0:i1],ids[j0:j1],length)
                if i0==j0:
                    np.fill_diagonal(dist,np.inf)
                    r1[i0:i1],r2[i0:i1]=self._merge(r1[i0:i1],r2[i0:i1],dist)
                else:
                    r1[i0:i1],r2[i0:i1]=self._merge(r1[i0:i1],r2[i0:i1],dist)
                    r1[j0:j1],r2[j0:j1]=self._merge(r1[j0:j1],r2[j0:j1],dist.T)
            if cp is not None:
                cp.parent.mkdir(parents=True,exist_ok=True)
                tmp=Path(str(cp)+'.tmp.npz')
                np.savez_compressed(tmp,r1=r1,r2=r2,identity=np.array(identity),block_size=B,next_outer=ai+1)
                os.replace(tmp,cp)
        if p is not None:
            p.parent.mkdir(parents=True,exist_ok=True)
            tmp=Path(str(p)+'.tmp.npz')
            np.savez_compressed(tmp,r1=r1,r2=r2,identity=np.array(identity),length=length,n=n)
            os.replace(tmp,p)
            with np.load(p,allow_pickle=False) as d:
                if str(d['identity'].item())!=identity: raise RuntimeError('Checkpoint verification failed')
            if cp is not None: cp.unlink(missing_ok=True)
        return r1,r2
