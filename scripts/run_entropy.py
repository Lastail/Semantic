#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from semantic.config import load_config
from semantic.generation.client import make_client
from semantic.metrics.entropy import configure_entropy,sweep_temperatures
ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/experiments/entropy.yaml"); args=ap.parse_args()
cfg=load_config(ROOT/"configs/default.yaml",ROOT/"configs/qwen32b.yaml",ROOT/args.config); tok=AutoTokenizer.from_pretrained(cfg["model"]["path"])
configure_entropy(make_client(cfg),cfg["model"]["api_name"],tok,cfg["entropy"].get("max_workers",50)); lohi=cfg["entropy"].get("temperature_range")
out=ROOT/cfg["project"]["output_dir"]/cfg["experiment"]["name"]/"entropy"; sweep_temperatures(ROOT/cfg["experiment"]["input"],str(out),cfg["entropy"].get("mode","both"),cfg["entropy"].get("max_texts"),cfg["entropy"].get("max_workers"),tuple(lohi) if lohi else None)
