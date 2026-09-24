import matplotlib.pyplot as plt

def plot_correlation(temperatures,values,path,ylabel="Correlation"):
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot(temperatures,values,"o-"); ax.set_xlabel("Temperature"); ax.set_ylabel(ylabel); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(path,bbox_inches="tight"); plt.close(fig)
