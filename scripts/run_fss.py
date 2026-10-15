#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from semantic.config import load_config
from semantic.pipeline import FSSPipeline

ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--model-config", required=True); args=ap.parse_args()
cfg=load_config(ROOT/"configs/default.yaml",ROOT/args.model_config,ROOT/args.config)
FSSPipeline(cfg).run()
