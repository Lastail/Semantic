#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic.config import load_config
from semantic.generation.sampler import collect_samples


def resolve_from_root(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def parse_temperatures(value: str) -> list[float]:
    temps = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not temps:
        raise argparse.ArgumentTypeError("No temperatures were provided.")
    return temps


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate text from an OpenAI-compatible model endpoint.")
    ap.add_argument("--model-config", default="configs/qwen32b.yaml",
                    help="Model YAML merged on top of configs/default.yaml.")
    ap.add_argument("--prompt", required=True, help="Prompt text file.")
    ap.add_argument("--out", required=True, help="Final JSON output path.")
    ap.add_argument("--temperatures", required=True,
                    help="Comma-separated temperatures, e.g. 0.0,0.2,0.4.")
    ap.add_argument("--samples", type=int, default=None,
                    help="Samples per temperature; overrides YAML.")
    ap.add_argument("--max-workers", type=int, default=None,
                    help="Concurrent API requests; overrides YAML.")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Maximum generated tokens; overrides YAML.")
    ap.add_argument("--retries", type=int, default=3,
                    help="Retries for each failed API request.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore an existing checkpoint and start from zero.")
    args = ap.parse_args()

    default_cfg = ROOT / "configs/default.yaml"
    model_cfg = resolve_from_root(args.model_config)
    prompt_path = resolve_from_root(args.prompt)
    out_path = resolve_from_root(args.out)

    cfg = load_config(default_cfg, model_cfg)

    if args.samples is not None:
        cfg.setdefault("generation", {})["samples_per_temperature"] = args.samples
    if args.max_workers is not None:
        cfg.setdefault("generation", {})["max_workers"] = args.max_workers
    if args.max_tokens is not None:
        cfg.setdefault("generation", {})["max_tokens"] = args.max_tokens

    prompt = prompt_path.read_text(encoding="utf-8")
    temps = parse_temperatures(args.temperatures)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_path.with_suffix(out_path.suffix + ".checkpoint.jsonl")
    metadata_path = out_path.with_suffix(out_path.suffix + ".metadata.json")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_config": str(model_cfg),
        "model_name": cfg.get("model", {}).get("name"),
        "api_name": cfg.get("model", {}).get("api_name"),
        "endpoint": cfg.get("model", {}).get("endpoint"),
        "prompt_file": str(prompt_path),
        "temperatures": temps,
        "samples_per_temperature": cfg["generation"]["samples_per_temperature"],
        "max_workers": cfg["generation"].get("max_workers", 50),
        "max_tokens": cfg["generation"]["max_tokens"],
        "thinking_enabled": cfg["generation"].get("thinking_enabled", False),
        "resume": not args.no_resume,
        "checkpoint_file": str(checkpoint_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    results = collect_samples(
        cfg=cfg,
        prompt=prompt,
        temperatures=temps,
        samples_per_temp=args.samples,
        checkpoint_path=checkpoint_path,
        resume=not args.no_resume,
        retries=args.retries,
    )

    # Keep the original downstream-compatible format:
    # [{"temperature": T, "texts": [...]}, ...]
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nDone: {out_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
