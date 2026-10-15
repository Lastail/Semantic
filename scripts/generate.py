#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# Project root
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from semantic.config import load_config
from semantic.generation.sampler import collect_samples


# ============================================================
# Utilities
# ============================================================

def resolve(path: str) -> Path:
    """
    相对路径统一相对于项目 ROOT 解析。
    """

    p = Path(path).expanduser()

    if p.is_absolute():
        return p

    return ROOT / p


def parse_temperatures(value: str) -> list[float]:
    """
    解析命令行：

        --temperatures 0.0,0.2,0.4

    """

    temps = [
        float(x.strip())
        for x in value.split(",")
        if x.strip()
    ]

    if not temps:
        raise argparse.ArgumentTypeError(
            "No temperatures provided."
        )

    return temps


def temperatures_from_config(
    generation_cfg: dict,
) -> list[float]:
    """
    从 YAML：

    generation:
      temperatures:
        start: 0.0
        stop: 3.0
        step: 0.2

    生成：

        [0.0, 0.2, ..., 3.0]
    """

    temp_cfg = generation_cfg.get(
        "temperatures"
    )

    if not temp_cfg:
        raise ValueError(
            "No temperatures specified.\n"
            "Either provide:\n"
            "  --temperatures 0.0,0.2,...\n"
            "or define:\n"
            "  generation.temperatures\n"
            "in YAML."
        )

    required = (
        "start",
        "stop",
        "step",
    )

    missing = [
        key
        for key in required
        if key not in temp_cfg
    ]

    if missing:
        raise ValueError(
            "Missing generation.temperatures "
            "settings: "
            + ", ".join(missing)
        )

    start = float(
        temp_cfg["start"]
    )

    stop = float(
        temp_cfg["stop"]
    )

    step = float(
        temp_cfg["step"]
    )

    if step <= 0:
        raise ValueError(
            "generation.temperatures.step "
            "must be > 0."
        )

    if stop < start:
        raise ValueError(
            "generation.temperatures.stop "
            "must be >= start."
        )

    # --------------------------------------------------------
    # 不直接使用 np.arange，避免：
    #
    # 0.6000000000000001
    # --------------------------------------------------------

    n_steps_float = (
        (stop - start)
        / step
    )

    n_steps = int(
        round(
            n_steps_float
        )
    )

    # 检查 stop 是否与 step 基本兼容
    reconstructed_stop = (
        start
        + n_steps * step
    )

    if abs(
        reconstructed_stop - stop
    ) > 1e-8:

        raise ValueError(
            "Temperature range is not evenly "
            "divisible by step:\n"
            f"start={start}, "
            f"stop={stop}, "
            f"step={step}"
        )

    temps = [
        round(
            start + i * step,
            10,
        )
        for i in range(
            n_steps + 1
        )
    ]

    return temps


def validate_model_config(
    cfg: dict,
) -> None:
    """
    防止 default.yaml / model YAML 混用模型配置。
    """

    model = cfg.get("model")

    if not model:
        raise ValueError(
            "No model configuration found. "
            "Please specify --model-config."
        )

    required = (
        "name",
        "path",
        "api_name",
        "endpoint",
    )

    missing = [
        key
        for key in required
        if not model.get(key)
    ]

    if missing:
        raise ValueError(
            "Incomplete model configuration: "
            + ", ".join(missing)
        )


def validate_generation_config(
    generation_cfg: dict,
) -> None:
    """
    检查 generation 必需参数。
    """

    required = (
        "samples_per_temperature",
        "max_workers",
        "max_tokens",
        "min_tokens",
    )

    missing = [
        key
        for key in required
        if key not in generation_cfg
    ]

    if missing:
        raise ValueError(
            "Missing generation settings: "
            + ", ".join(missing)
        )

    samples = int(
        generation_cfg[
            "samples_per_temperature"
        ]
    )

    workers = int(
        generation_cfg[
            "max_workers"
        ]
    )

    max_tokens = int(
        generation_cfg[
            "max_tokens"
        ]
    )

    min_tokens = int(
        generation_cfg[
            "min_tokens"
        ]
    )

    if samples <= 0:
        raise ValueError(
            "samples_per_temperature "
            "must be > 0."
        )

    if workers <= 0:
        raise ValueError(
            "max_workers must be > 0."
        )

    if max_tokens <= 0:
        raise ValueError(
            "max_tokens must be > 0."
        )

    if min_tokens < 0:
        raise ValueError(
            "min_tokens must be >= 0."
        )

    if min_tokens > max_tokens:
        print(
            "[warning] "
            f"min_tokens={min_tokens} > "
            f"max_tokens={max_tokens}."
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # CLI
    # --------------------------------------------------------

    ap = argparse.ArgumentParser(
        description=(
            "Generate temperature-controlled "
            "text samples."
        )
    )

    ap.add_argument(
        "--model-config",
        required=True,
        help=(
            "Model YAML, e.g. "
            "configs/qwen06b.yaml"
        ),
    )

    ap.add_argument(
        "--prompt",
        required=True,
        help="Prompt text file.",
    )

    ap.add_argument(
        "--out",
        required=True,
        help="Final dataset JSON.",
    )

    # --------------------------------------------------------
    # Optional overrides
    # --------------------------------------------------------

    ap.add_argument(
        "--temperatures",
        default=None,
        help=(
            "Optional comma-separated "
            "temperature list. "
            "If omitted, use "
            "generation.temperatures "
            "from YAML."
        ),
    )

    ap.add_argument(
        "--samples",
        type=int,
        default=None,
        help=(
            "Override "
            "samples_per_temperature."
        ),
    )

    ap.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Override max_workers.",
    )

    ap.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override max_tokens.",
    )

    ap.add_argument(
        "--min-tokens",
        type=int,
        default=None,
        help=(
            "Override min_tokens. "
            "Shorter outputs are rejected."
        ),
    )

    ap.add_argument(
        "--retries",
        type=int,
        default=3,
        help=(
            "Retries for individual "
            "failed API requests."
        ),
    )

    ap.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Ignore/delete the existing "
            "checkpoint for this output."
        ),
    )

    args = ap.parse_args()

    # ========================================================
    # Load configuration
    # ========================================================

    default_cfg = (
        ROOT
        / "configs"
        / "default.yaml"
    )

    model_cfg = resolve(
        args.model_config
    )

    cfg = load_config(
        default_cfg,
        model_cfg,
    )

    validate_model_config(
        cfg
    )

    generation = cfg.setdefault(
        "generation",
        {},
    )

    # ========================================================
    # CLI overrides YAML
    # ========================================================

    overrides = {

        "samples_per_temperature":
            args.samples,

        "max_workers":
            args.max_workers,

        "max_tokens":
            args.max_tokens,

        "min_tokens":
            args.min_tokens,
    }

    for key, value in overrides.items():

        if value is not None:
            generation[key] = value

    validate_generation_config(
        generation
    )

    # ========================================================
    # Temperature grid
    # ========================================================

    if args.temperatures is not None:

        temps = parse_temperatures(
            args.temperatures
        )

        temperature_source = (
            "command line"
        )

    else:

        temps = temperatures_from_config(
            generation
        )

        temperature_source = (
            "YAML"
        )

    # ========================================================
    # Prompt / output
    # ========================================================

    prompt_path = resolve(
        args.prompt
    )

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt not found: "
            f"{prompt_path}"
        )

    prompt = prompt_path.read_text(
        encoding="utf-8"
    )

    out_path = resolve(
        args.out
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        out_path.with_suffix(
            out_path.suffix
            + ".checkpoint.jsonl"
        )
    )

    metadata_path = (
        out_path.with_suffix(
            out_path.suffix
            + ".metadata.json"
        )
    )

    # ========================================================
    # Metadata
    # ========================================================

    metadata = {

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "model_config":
            str(model_cfg),

        "model_name":
            cfg["model"]["name"],

        "model_path":
            cfg["model"]["path"],

        "api_name":
            cfg["model"]["api_name"],

        "endpoint":
            cfg["model"]["endpoint"],

        "prompt_file":
            str(prompt_path),

        "temperature_source":
            temperature_source,

        "temperatures":
            temps,

        "samples_per_temperature":
            int(
                generation[
                    "samples_per_temperature"
                ]
            ),

        "max_workers":
            int(
                generation[
                    "max_workers"
                ]
            ),

        "max_tokens":
            int(
                generation[
                    "max_tokens"
                ]
            ),

        "min_tokens":
            int(
                generation[
                    "min_tokens"
                ]
            ),

        "thinking_enabled":
            bool(
                generation.get(
                    "thinking_enabled",
                    False,
                )
            ),

        "resume":
            not args.no_resume,

        "checkpoint_file":
            str(checkpoint_path),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Print resolved configuration
    # ========================================================

    print("=" * 80)
    print("Generation configuration")
    print("=" * 80)

    print(
        f"model          : "
        f"{metadata['model_name']}"
    )

    print(
        f"model path     : "
        f"{metadata['model_path']}"
    )

    print(
        f"endpoint       : "
        f"{metadata['endpoint']}"
    )

    print(
        f"prompt         : "
        f"{prompt_path}"
    )

    print(
        f"samples / T    : "
        f"{metadata['samples_per_temperature']}"
    )

    print(
        f"max tokens     : "
        f"{metadata['max_tokens']}"
    )

    print(
        f"min tokens     : "
        f"{metadata['min_tokens']}"
    )

    print(
        f"workers        : "
        f"{metadata['max_workers']}"
    )

    print(
        f"thinking       : "
        f"{metadata['thinking_enabled']}"
    )

    print(
        f"temperatures   : "
        f"{temps}"
    )

    print(
        f"temperature src: "
        f"{temperature_source}"
    )

    print(
        f"resume         : "
        f"{metadata['resume']}"
    )

    print("=" * 80)

    # ========================================================
    # Generate
    # ========================================================

    results = collect_samples(
        cfg=cfg,
        prompt=prompt,
        temperatures=temps,
        checkpoint_path=checkpoint_path,
        resume=not args.no_resume,
        retries=args.retries,
    )

    # ========================================================
    # Final dataset
    # ========================================================

    out_path.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Done       : {out_path}"
    )

    print(
        f"Metadata   : {metadata_path}"
    )

    print(
        f"Checkpoint : {checkpoint_path}"
    )


if __name__ == "__main__":
    main()