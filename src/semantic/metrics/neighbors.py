import numpy as np
from sklearn.neighbors import NearestNeighbors

def euclidean_two_nearest(vectors):
    x=np.asarray(vectors,dtype=np.float32)
    if x.ndim!=2 or len(x)<3:
        raise ValueError(f"Need [N,D], N>=3; got {x.shape}")
    nn=NearestNeighbors(n_neighbors=3,metric="euclidean",n_jobs=-1).fit(x)
    d,_=nn.kneighbors(x)
    return d[:,1].astype(np.float64),d[:,2].astype(np.float64)
