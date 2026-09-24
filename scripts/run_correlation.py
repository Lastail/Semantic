#!/usr/bin/env python3
import argparse,sys,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from semantic.metrics.correlation import mean_pairwise_cosine
ap=argparse.ArgumentParser(); ap.add_argument("npz"); args=ap.parse_args(); d=np.load(args.npz); print(mean_pairwise_cosine(d["vecs"]))
