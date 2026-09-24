#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from semantic.config import load_config
from semantic.generation.sampler import collect_samples
ap=argparse.ArgumentParser(); ap.add_argument("--prompt",required=True); ap.add_argument("--out",required=True); ap.add_argument("--temperatures",required=True); args=ap.parse_args()
cfg=load_config(ROOT/"configs/default.yaml",ROOT/"configs/qwen32b.yaml"); prompt=Path(args.prompt).read_text(encoding="utf-8"); temps=[float(x) for x in args.temperatures.split(",")]
out=collect_samples(cfg,prompt,temps); p=ROOT/args.out; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
