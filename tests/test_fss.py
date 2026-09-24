import numpy as np
from semantic.metrics.two_nn import compute_intrinsic_dimensions_from_vectors

def test_grouping():
    rng=np.random.default_rng(1); v=rng.normal(size=(20,4)); t=np.repeat([1.0,1.2],10); out,_=compute_intrinsic_dimensions_from_vectors(v,t,100,None); assert 100 in out and len(out[100]["temps"])==2
