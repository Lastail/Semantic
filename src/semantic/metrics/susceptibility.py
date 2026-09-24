import numpy as np

def susceptibility_from_token_embeddings(samples):
    """samples: iterable of arrays [tokens, dim], reproduces <m^2>-<m>^2 style statistic."""
    samples=[np.asarray(x,float) for x in samples if len(x)]
    if not samples:return 0.0
    L=min(len(x) for x in samples); xs=[x[:L] for x in samples]
    sus1=np.mean([np.sum(x@x.T)/L for x in xs]); avg=np.mean(xs,axis=0); sus2=np.sum(avg@avg.T)/L
    return float(sus1-sus2)
