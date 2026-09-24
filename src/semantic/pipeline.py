from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .data.loader import load_temperature_groups
from .data.preprocessing import split_into_blocks
from .representations import build_extractor
from .cache.checkpoint import checkpoint_path,load_checkpoint,atomic_save
from .cache.metadata import save_metadata
from .metrics.two_nn import compute_intrinsic_dimensions_from_vectors
from .fss.collapse import perform_data_collapse
from .fss.tstar import plot_Tstar_multimodel
from .fss.fitting import plot_fit_subplots_from_fitdata
from .plotting.intrinsic_dimension import plot_id_vs_temperature

class FSSPipeline:
    def __init__(self,cfg): self.cfg=cfg; self.extractor=None
    def _experiment_dir(self):
        name=self.cfg.get("experiment",{}).get("name","experiment")
        p=Path(self.cfg["project"]["output_dir"])/name; p.mkdir(parents=True,exist_ok=True); return p
    def _prepare_texts(self,texts,size):
        if self.cfg["fss"].get("size_type","token")=="block": return [" ".join(split_into_blocks(t)[:size]) for t in texts]
        return texts
    def compute_vectors(self,data_path):
        groups=load_temperature_groups(data_path); outdir=self._experiment_dir(); cache=outdir/"vectors"; cache.mkdir(parents=True,exist_ok=True)
        self.extractor=self.extractor or build_extractor(self.cfg); paths={}
        for size in self.cfg["fss"]["sizes"]:
            all_v=[]; all_t=[]
            for T in sorted(groups):
                p=checkpoint_path(cache,T,size); ck=load_checkpoint(p,T)
                if ck is None:
                    texts=self._prepare_texts(groups[T],size)
                    max_tokens=size if self.cfg["fss"].get("size_type","token")=="token" else 10**9
                    v=self.extractor.encode(texts,max_tokens)
                    if len(v)<3: continue
                    ts=np.full(len(v),T,dtype=np.float64); atomic_save(p,v,ts)
                else: v,ts=ck
                all_v.append(v); all_t.append(ts)
            if all_v:
                final=cache/f"vectors_S{size}.npz"; atomic_save(final,np.concatenate(all_v),np.concatenate(all_t)); paths[size]=final
        return paths
    def run(self):
        data_path=self.cfg["experiment"]["input"]; outdir=self._experiment_dir(); save_metadata(outdir/"metadata.json",self.cfg)
        paths=self.compute_vectors(data_path); outcomes={}; fits={}
        for size,p in paths.items():
            with np.load(p) as d: out,fit=compute_intrinsic_dimensions_from_vectors(d["vecs"],d["temps"],size,self.cfg["fss"].get("mu_cutoff"))
            outcomes.update(out); fits.update(fit)
        lo,hi=self.cfg["fss"].get("temperature_range",[-np.inf,np.inf]); filtered={}
        for size,d in outcomes.items():
            t=np.asarray(d["temps"]); m=(t>=lo)&(t<=hi)
            if m.any(): filtered[size]={k:(np.asarray(v)[m].tolist() if k in ("temps","ids","stds") else v) for k,v in d.items()}
        outcomes=filtered
        (outdir/"intrinsic_dimension.json").write_text(json.dumps(outcomes,indent=2),encoding="utf-8")
        plot_id_vs_temperature(outcomes,outdir/"id_vs_temperature.pdf", "B" if self.cfg["fss"].get("size_type")=="block" else "L")
        f=self.cfg["fss"]; collapse=perform_data_collapse(outcomes,str(outdir/"collapse.pdf"),f.get("guess_Tc",1.2),f.get("guess_nu",1.0),f.get("guess_xi",0.0))
        if collapse and collapse.get("Tc") is not None: plot_Tstar_multimodel(outcomes,collapse.get("Tc"),collapse.get("nu"),d=2,output_dir=str(outdir))
        if fits: plot_fit_subplots_from_fitdata(fits,str(outdir),model_label=self.cfg["representation"]["type"],size_label="B" if f.get("size_type")=="block" else "L")
        return collapse
