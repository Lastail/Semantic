import numpy as np
from .neighbors import euclidean_two_nearest

def fit_two_nn_from_neighbors(r1,r2,mu_cutoff=0.2,discard_fraction=0,return_fit_data=False):
    r1=np.asarray(r1,dtype=np.float64); r2=np.asarray(r2,dtype=np.float64)
    if r1.shape!=r2.shape or r1.ndim!=1 or len(r1)<3:
        raise ValueError(f"Invalid r1/r2: {r1.shape}, {r2.shape}")
    finite=np.isfinite(r1)&np.isfinite(r2)
    zero_r1=int(np.sum(r1[finite]==0)); zero_r2=int(np.sum(r2[finite]==0))
    r1,r2=r1[finite],r2[finite]
    if len(r1)<3: raise ValueError("Fewer than 3 finite TWO-NN samples")
    eps=1e-12
    r1=np.where(r1==0,eps,r1); r2=np.where(r2==0,eps,r2)
    mu=np.maximum((r2/r1)[np.isfinite(r2/r1)],1.0)
    mu=np.sort(mu)
    mu=mu[:int(len(mu)*(1-discard_fraction))]
    if len(mu)<2: raise ValueError("Not enough TWO-NN ratios")
    F=np.arange(1,len(mu)+1)/(len(mu)+1.0)
    x=np.log(mu); y=-np.log(1-F)
    if mu_cutoff is not None:
        m=x<=float(mu_cutoff)
        if m.sum()>=2: x,y=x[m],y[m]
    den=np.sum(x*x)
    if den<=0 or len(x)<2: raise ValueError("Degenerate TWO-NN fit")
    slope=float(np.sum(x*y)/den)
    mse=float(np.sum((y-slope*x)**2)/(len(x)-1))
    std=float(np.sqrt(mse/den))
    diag={"n_finite":int(finite.sum()),"n_fit":len(x),"zero_r1":zero_r1,"zero_r2":zero_r2}
    return (slope,std,x,y,diag) if return_fit_data else slope

def intrinsic_dimension_two_nn(points,discard_fraction=0,metric="euclidean",n_jobs=-1,return_fit_data=False,mu_cutoff=None):
    if metric!="euclidean": raise ValueError("Only euclidean supported")
    r1,r2=euclidean_two_nearest(points)
    slope,std,x,y,_=fit_two_nn_from_neighbors(r1,r2,mu_cutoff,discard_fraction,True)
    return (slope,std,x,y) if return_fit_data else slope

def compute_intrinsic_dimensions_from_vectors(vecs,temps,size,mu_cutoff=0.2):
    vecs=np.asarray(vecs); temps=np.asarray(temps)
    ts=[]; ids=[]; stds=[]; fit=[]
    for t in np.unique(temps):
        g=vecs[temps==t]
        if len(g)<3: continue
        r1,r2=euclidean_two_nearest(g)
        slope,std,x,y,diag=fit_two_nn_from_neighbors(r1,r2,mu_cutoff,return_fit_data=True)
        if diag["zero_r1"]: print(f"[TWO-NN] T={t:g}, L={size}: zero-r1={diag['zero_r1']}")
        ts.append(float(t)); ids.append(slope); stds.append(std); fit.append((float(t),slope,x.copy(),y.copy()))
    out={}; fits={}
    if ts:
        idx=np.argsort(ts)
        out[size]={"temps":[ts[i] for i in idx],"ids":[ids[i] for i in idx],"stds":[stds[i] for i in idx]}
        fit.sort(key=lambda z:z[0]); fits[size]=fit
    return out,fits
