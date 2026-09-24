import numpy as np
from semantic.metrics.two_nn import intrinsic_dimension_two_nn

def test_two_nn_finite():
    rng=np.random.default_rng(0); x=rng.normal(size=(200,3)); d=intrinsic_dimension_two_nn(x); assert np.isfinite(d) and d>0
