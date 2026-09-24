from pathlib import Path
import os, numpy as np

def temperature_tag(t): return f"{float(t):.12g}".replace("-","m").replace(".","p")
def checkpoint_path(root,T,size): return Path(root)/"checkpoints"/f"vectors_T{temperature_tag(T)}_S{int(size)}.npz"
def load_checkpoint(path, expected_T=None):
    p=Path(path)
    if not p.exists(): return None
    try:
        with np.load(p,allow_pickle=False) as d: v=np.asarray(d["vecs"],np.float32); t=np.asarray(d["temps"],np.float64)
        if v.ndim!=2 or t.ndim!=1 or len(v)!=len(t) or not len(v) or not np.isfinite(v).all(): return None
        if expected_T is not None and not np.allclose(t,expected_T,rtol=0,atol=1e-10): return None
        return v,t
    except Exception: return None
def atomic_save(path,vecs,temps):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); tmp=Path(str(p)+".tmp.npz")
    np.savez_compressed(tmp,vecs=np.asarray(vecs,np.float32),temps=np.asarray(temps,np.float64))
    if load_checkpoint(tmp) is None: raise RuntimeError(f"checkpoint validation failed: {tmp}")
    os.replace(tmp,p)
