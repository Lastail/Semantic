import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    path = Path(args.input)
    data = json.loads(path.read_text())

    lengths = sorted(set(int(x["length"]) for x in data))

    fig, ax = plt.subplots(figsize=(8, 6))

    for L in lengths:
        rows = sorted(
            (x for x in data if int(x["length"]) == L),
            key=lambda x: float(x["temperature"])
        )

        temps = [float(x["temperature"]) for x in rows]
        chi = [float(x["susceptibility"]) for x in rows]

        ax.plot(temps, chi, "o-", label=f"L={L}")

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Susceptibility")
    ax.legend()
    ax.grid(alpha=0.3)

    output = path.parent / "susceptibility_vs_temperature.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output}")

if __name__ == "__main__":
    main()