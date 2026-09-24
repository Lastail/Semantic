import numpy as np

def l2_normalize_rows(x,eps=1e-12):
    x=np.asarray(x,float); n=np.linalg.norm(x,axis=1,keepdims=True); ok=n[:,0]>eps; return x[ok]/n[ok]
def mean_pairwise_cosine(vectors):
    x=l2_normalize_rows(vectors); n=len(x)
    if n<2:return float("nan"),float("nan")
    vals=(x@x.T)[np.triu_indices(n,1)]
    return float(vals.mean()),float(vals.std())
