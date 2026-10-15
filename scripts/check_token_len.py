#!/usr/bin/env python3

import argparse
import json
import numpy as np

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="检查生成文本的 token 长度分布"
    )

    parser.add_argument(
        "--data",
        required=True,
        help="生成数据 JSON 文件",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Tokenizer 模型路径",
    )

    parser.add_argument(
        "--thresholds",
        default="100,500,1000,2000,3000,3500,4000",
        help="需要统计的 token 长度阈值",
    )

    parser.add_argument(
        "--show-short",
        type=int,
        default=3,
        help="每个温度显示多少个 <3000 token 的样本",
    )

    args = parser.parse_args()

    thresholds = [
        int(x)
        for x in args.thresholds.split(",")
    ]

    # ============================================================
    # tokenizer
    # ============================================================

    print("=" * 100)
    print("Loading tokenizer")
    print("=" * 100)

    print("model:", args.model)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    print(
        "tokenizer:",
        tokenizer.__class__.__name__,
    )

    # ============================================================
    # load data
    # ============================================================

    print()
    print("=" * 100)
    print("Loading data")
    print("=" * 100)

    print("data:", args.data)

    with open(
        args.data,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "JSON 顶层应该是 list"
        )

    # ============================================================
    # header
    # ============================================================

    print()
    print("=" * 150)
    print("TOKEN LENGTH STATISTICS")
    print("=" * 150)

    header = (
        f"{'T':>6}"
        f"{'N':>7}"
        f"{'mean':>10}"
        f"{'median':>10}"
        f"{'std':>10}"
        f"{'min':>8}"
        f"{'p10':>8}"
        f"{'p25':>8}"
        f"{'p75':>8}"
        f"{'p90':>8}"
        f"{'max':>8}"
    )

    for threshold in thresholds:

        header += (
            f"{'>=' + str(threshold):>10}"
        )

    print(header)

    print("-" * len(header))

    all_results = {}

    # ============================================================
    # each temperature
    # ============================================================

    for item in data:

        T = float(
            item["temperature"]
        )

        texts = item["texts"]

        lengths = []

        # --------------------------------------------------------
        # tokenize
        # --------------------------------------------------------

        for text in texts:

            ids = tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]

            lengths.append(
                len(ids)
            )

        lengths = np.asarray(
            lengths,
            dtype=np.int64,
        )

        if len(lengths) == 0:
            continue

        # --------------------------------------------------------
        # statistics
        # --------------------------------------------------------

        result = {

            "N":
                len(lengths),

            "mean":
                float(
                    np.mean(lengths)
                ),

            "median":
                float(
                    np.median(lengths)
                ),

            "std":
                float(
                    np.std(lengths)
                ),

            "min":
                int(
                    np.min(lengths)
                ),

            "max":
                int(
                    np.max(lengths)
                ),

            "p10":
                float(
                    np.percentile(
                        lengths,
                        10,
                    )
                ),

            "p25":
                float(
                    np.percentile(
                        lengths,
                        25,
                    )
                ),

            "p75":
                float(
                    np.percentile(
                        lengths,
                        75,
                    )
                ),

            "p90":
                float(
                    np.percentile(
                        lengths,
                        90,
                    )
                ),
        }

        for threshold in thresholds:

            result[
                f">={threshold}"
            ] = int(
                np.sum(
                    lengths
                    >= threshold
                )
            )

        all_results[T] = result

        # --------------------------------------------------------
        # print row
        # --------------------------------------------------------

        row = (
            f"{T:6.2f}"
            f"{len(lengths):7d}"
            f"{np.mean(lengths):10.1f}"
            f"{np.median(lengths):10.1f}"
            f"{np.std(lengths):10.1f}"
            f"{np.min(lengths):8d}"
            f"{np.percentile(lengths,10):8.0f}"
            f"{np.percentile(lengths,25):8.0f}"
            f"{np.percentile(lengths,75):8.0f}"
            f"{np.percentile(lengths,90):8.0f}"
            f"{np.max(lengths):8d}"
        )

        for threshold in thresholds:

            count = np.sum(
                lengths
                >= threshold
            )

            row += (
                f"{count:10d}"
            )

        print(row)

    # ============================================================
    # 找 <3000 的样本
    # ============================================================

    print()
    print("=" * 100)
    print("EXAMPLES WITH < 3000 TOKENS")
    print("=" * 100)

    for item in data:

        T = float(
            item["temperature"]
        )

        texts = item["texts"]

        shown = 0

        for index, text in enumerate(
            texts
        ):

            ids = tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]

            if len(ids) >= 3000:
                continue

            print()
            print("-" * 100)

            print(
                f"T={T:.2f}"
                f" | sample={index}"
                f" | tokens={len(ids)}"
                f" | chars={len(text)}"
            )

            print(
                "Last 300 characters:"
            )

            print(
                repr(
                    text[-300:]
                )
            )

            # ----------------------------------------------------
            # 最后 1000 token 的多样性
            # ----------------------------------------------------

            tail = ids[-1000:]

            unique_tail = len(
                set(tail)
            )

            print(
                "unique token IDs "
                f"in last {len(tail)} tokens:"
                f" {unique_tail}"
            )

            shown += 1

            if shown >= args.show_short:
                break

    # ============================================================
    # repetition diagnostic
    # ============================================================

    print()
    print("=" * 100)
    print("REPETITION DIAGNOSTIC")
    print("=" * 100)

    print(
        f"{'T':>6}"
        f"{'N':>7}"
        f"{'tail_unique_mean':>20}"
        f"{'tail_unique_median':>22}"
        f"{'<=10 unique':>15}"
        f"{'<=20 unique':>15}"
    )

    for item in data:

        T = float(
            item["temperature"]
        )

        unique_counts = []

        for text in item["texts"]:

            ids = tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]

            tail = ids[-1000:]

            unique_counts.append(
                len(
                    set(tail)
                )
            )

        unique_counts = np.asarray(
            unique_counts
        )

        print(
            f"{T:6.2f}"
            f"{len(unique_counts):7d}"
            f"{unique_counts.mean():20.1f}"
            f"{np.median(unique_counts):22.1f}"
            f"{np.sum(unique_counts <= 10):15d}"
            f"{np.sum(unique_counts <= 20):15d}"
        )

    print()
    print("=" * 100)
    print("Done")
    print("=" * 100)


if __name__ == "__main__":
    main()