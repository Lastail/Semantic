from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from tqdm import tqdm

from .client import make_client


def generate_text(client, cfg, prompt, temperature, retries=3):
    """Generate one sample, retrying transient API failures."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            r = client.chat.completions.create(
                model=cfg["model"]["api_name"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=cfg["generation"]["max_tokens"],
                temperature=temperature,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": cfg["generation"].get(
                            "thinking_enabled", False
                        )
                    }
                },
            )
            return r.choices[0].message.content
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(
        f"Generation failed at T={temperature} after {retries + 1} attempts"
    ) from last_error


def _load_checkpoint(path: Path) -> dict[float, list[str]]:
    """Read successful samples from an append-only JSONL checkpoint."""
    recovered: dict[float, list[str]] = {}
    if not path.exists():
        return recovered

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                T = float(row["temperature"])
                text = row["text"]
                if isinstance(text, str):
                    recovered.setdefault(T, []).append(text)
            except Exception as exc:
                print(f"[warning] skip malformed checkpoint line {line_no}: {exc}")
    return recovered


def collect_samples(
    cfg,
    prompt,
    temperatures,
    samples_per_temp=None,
    checkpoint_path=None,
    resume=True,
    retries=3,
):
    """
    Generate samples for each temperature.

    Successful samples are appended immediately to a JSONL checkpoint, so an
    interrupted run can resume without regenerating completed samples.
    """
    client = make_client(cfg)
    n = (
        samples_per_temp
        if samples_per_temp is not None
        else cfg["generation"]["samples_per_temperature"]
    )
    max_workers = cfg["generation"].get("max_workers", 50)

    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint is not None and not resume and checkpoint.exists():
        checkpoint.unlink()

    recovered = _load_checkpoint(checkpoint) if (checkpoint and resume) else {}
    write_lock = Lock()
    out = []

    for T_raw in temperatures:
        T = float(T_raw)
        texts = list(recovered.get(T, []))

        # If an old checkpoint contains more samples than currently requested,
        # use only the requested number in the final dataset.
        if len(texts) >= n:
            print(f"T={T:g}: resume {n}/{n}, skip generation")
            out.append({"temperature": T, "texts": texts[:n]})
            continue

        remaining = n - len(texts)
        print(f"T={T:g}: recovered {len(texts)}/{n}; generating {remaining}")

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(generate_text, client, cfg, prompt, T, retries)
                for _ in range(remaining)
            ]

            for future in tqdm(
                as_completed(futures),
                total=remaining,
                desc=f"T={T:g}",
            ):
                text = future.result()
                texts.append(text)

                # Append every successful sample immediately.
                if checkpoint is not None:
                    row = json.dumps(
                        {"temperature": T, "text": text},
                        ensure_ascii=False,
                    )
                    with write_lock:
                        with checkpoint.open("a", encoding="utf-8") as f:
                            f.write(row + "\n")
                            f.flush()

        out.append({"temperature": T, "texts": texts[:n]})

    return out
