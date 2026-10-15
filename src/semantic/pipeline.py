from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from .data.loader import load_temperature_groups
from .representations import build_representation
from .cache.metadata import save_metadata
from .metrics.two_nn import fit_two_nn_from_neighbors
from .fss.collapse import perform_data_collapse
from .fss.tstar import plot_Tstar_multimodel
from .fss.fitting import plot_fit_subplots_from_fitdata
from .plotting.intrinsic_dimension import plot_id_vs_temperature

def _tag(t):
    return f"{float(t):.12g}".replace("-","m").replace(".","p")

class FSSPipeline:
    def __init__(self,cfg):
        self.cfg=cfg
        self.representation=None

    def _experiment_dir(self):
        explicit=self.cfg.get("output",{}).get("dir")
        if explicit:
            p=Path(explicit)
        else:
            name=self.cfg.get("experiment",{}).get("name","experiment")
            p=Path(self.cfg.get("project",{}).get("output_dir","outputs"))/name
        p.mkdir(parents=True,exist_ok=True)
        return p

    def _common_ensemble(self,groups):
        """Optional: same texts for all L by requiring max(size) tokens."""
        if not self.cfg["fss"].get("common_ensemble",False):
            return groups
        max_L=max(map(int,self.cfg["fss"]["sizes"]))
        tok=self.representation.tokenizer
        out={}
        for T,texts in groups.items():
            kept=[]
            for text in texts:
                ids=tok(text,add_special_tokens=True,truncation=False)["input_ids"]
                if len(ids)>=max_L: kept.append(text)
            print(f"[common ensemble] T={T:g}: {len(kept)}/{len(texts)} texts >= {max_L} tokens")
            out[T]=kept
        return out

    def run(self):
        data_path=self.cfg["experiment"]["input"]
        outdir=self._experiment_dir()
        save_metadata(outdir/"metadata.json",self.cfg)

        groups=load_temperature_groups(data_path)
        self.representation=build_representation(self.cfg)
        groups=self._common_ensemble(groups)

        fss=self.cfg["fss"]
        sizes=[int(x) for x in fss["sizes"]]
        mu_cutoff=fss.get("mu_cutoff",0.2)
        lo,hi=fss.get("temperature_range",[-np.inf,np.inf])

        outcomes={}
        fits={}
        diagnostics={}
        neighbor_dir=outdir/"neighbors"
        neighbor_dir.mkdir(parents=True,exist_ok=True)

        for L in sizes:
            ts=[]; ids=[]; stds=[]; fit_list=[]
            diagnostics[L]={}

            for T in sorted(groups):
                if not (lo<=T<=hi):
                    continue
                texts=groups[T]
                cache=neighbor_dir/f"{self.cfg['representation']['type']}_T{_tag(T)}_L{L}.npz"

                try:
                    r1,r2=self.representation.compute_neighbors(texts,L,cache)
                    slope,std,x,y,diag=fit_two_nn_from_neighbors(
                        r1,r2,mu_cutoff=mu_cutoff,return_fit_data=True
                    )
                except ValueError as e:
                    print(f"[skip] T={T:g}, L={L}: {e}")
                    continue

                print(f"[ID] T={T:g}, L={L}: d={slope:.6f} ± {std:.6f}, Nfit={diag['n_fit']}, zero-r1={diag['zero_r1']}")
                ts.append(float(T)); ids.append(slope); stds.append(std)
                fit_list.append((float(T),slope,x.copy(),y.copy()))
                diagnostics[L][str(T)]=diag

            if ts:
                idx=np.argsort(ts)
                outcomes[L]={"temps":[ts[i] for i in idx],"ids":[ids[i] for i in idx],"stds":[stds[i] for i in idx]}
                fit_list.sort(key=lambda z:z[0]); fits[L]=fit_list

        if not outcomes:
            raise RuntimeError("No valid intrinsic-dimension results")

        (outdir/"intrinsic_dimension.json").write_text(json.dumps(outcomes,indent=2),encoding="utf-8")
        (outdir/"two_nn_diagnostics.json").write_text(json.dumps(diagnostics,indent=2),encoding="utf-8")

        plot_id_vs_temperature(outcomes,outdir/"id_vs_temperature.pdf","L")

        guess=self.cfg.get("initial_guess",{})
        # Backward compatibility with older fss.guess_* configs.
        Tc=guess.get("Tc",fss.get("guess_Tc",1.2))
        nu=guess.get("nu",fss.get("guess_nu",1.0))
        xi=guess.get("xi",fss.get("guess_xi",0.0))

        collapse=perform_data_collapse(outcomes,str(outdir/"collapse.pdf"),Tc,nu,xi)

        if collapse and collapse.get("Tc") is not None:
            plot_Tstar_multimodel(outcomes,collapse.get("Tc"),collapse.get("nu"),d=2,output_dir=str(outdir))

        if fits:
            plot_fit_subplots_from_fitdata(fits,str(outdir),model_label=self.cfg["representation"]["type"],size_label="L")

        return collapse
