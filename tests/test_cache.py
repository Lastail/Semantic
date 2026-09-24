import numpy as np
from semantic.cache.checkpoint import atomic_save,load_checkpoint

def test_checkpoint(tmp_path):
    p=tmp_path/"x.npz"; v=np.ones((3,4),np.float32); t=np.ones(3); atomic_save(p,v,t); r=load_checkpoint(p,1.0); assert r is not None and r[0].shape==(3,4)
