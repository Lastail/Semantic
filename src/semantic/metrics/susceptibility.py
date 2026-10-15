"""Connected positional susceptibility, population ensemble convention."""
import numpy as np


def susceptibility_from_token_embeddings(samples, length=None):
    """Samples: iterable of [tokens, dim] arrays; all must have >= length tokens.

    chi = (1/L) sum_ij [<e_i.e_j> - <e_i>.<e_j>].
    Uses algebraically identical vector-sum identity, no LxL matrix.
    """
    xs = [np.asarray(x, dtype=np.float64) for x in samples]
    if not xs:
        raise ValueError('No samples')
    if length is None:
        length = min(x.shape[0] for x in xs)
    if length <= 0 or any(x.ndim != 2 or len(x) < length for x in xs):
        raise ValueError('Samples must be 2D and contain at least L tokens')
    if len({x.shape[1] for x in xs}) != 1:
        raise ValueError('Embedding dimensions differ')
    sums = np.stack([x[:length].sum(axis=0) for x in xs])
    if not np.isfinite(sums).all():
        raise ValueError('Nonfinite embeddings')
    center = sums.mean(axis=0)
    return float(np.square(sums - center).sum() / (len(xs) * length))
