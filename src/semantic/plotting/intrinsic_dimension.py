from pathlib import Path
import matplotlib.pyplot as plt

def plot_id_vs_temperature(outcomes,path,size_label="L"):
    fig,ax=plt.subplots(figsize=(10,6))
    for size in sorted(outcomes):
        d=outcomes[size]; ax.errorbar(d["temps"],d["ids"],yerr=d["stds"],marker="o",capsize=3,label=f"{size_label}={size}")
    ax.set_xlabel("Temperature"); ax.set_ylabel("Intrinsic Dimension"); ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    Path(path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,bbox_inches="tight"); plt.close(fig)
