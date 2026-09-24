from __future__ import annotations
import json
from collections import defaultdict


def load_temperature_entries(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("results_prompt", "two_nn_results", "results"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("数据必须是 [{temperature, texts}, ...] 或包含 results_prompt/two_nn_results/results 的 dict")
    return data


def load_temperature_groups(path):
    groups = defaultdict(list)
    for item in load_temperature_entries(path):
        groups[float(item["temperature"])].extend(item["texts"])
    return dict(groups)
