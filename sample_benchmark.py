"""
Sample a stratified benchmark dataset for consistent evaluation.

Uses the deduplicated test split (one doc per unique BDRC work) and
optionally sub-samples by boundary density strata for a compact benchmark.

Usage:
    python sample_benchmark.py
    python sample_benchmark.py --num-docs 20 --max-windows 200
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_from_disk

from config import (
    ANNOTATIONS_FILE,
    BENCHMARK_DIR,
    DOCUMENTS_DIR,
    PROCESSED_DIR,
    SEED,
)


def load_test_doc_ids() -> list[str]:
    split_info_path = PROCESSED_DIR / "split_info.json"
    if not split_info_path.exists():
        raise FileNotFoundError(
            f"{split_info_path} not found. Run prepare_data.py first."
        )
    with open(split_info_path) as f:
        split_info = json.load(f)
    return split_info["test_doc_ids"]


def load_doc_stats(doc_ids: list[str], annotations: dict) -> list[dict]:
    stats = []
    for doc_id in doc_ids:
        ann = annotations.get(doc_id)
        if ann is None:
            continue
        doc_path = DOCUMENTS_DIR / f"{doc_id}.txt"
        if not doc_path.exists():
            continue
        text_len = doc_path.stat().st_size
        n_bp = len(ann["breakpoints"])
        density = n_bp / max(text_len, 1) * 10000
        stats.append(
            {
                "doc_id": doc_id,
                "work_id": ann["filename"].split("_", 1)[0],
                "filename": ann["filename"],
                "text_length": text_len,
                "num_breakpoints": n_bp,
                "density": density,
            }
        )
    return stats


def stratified_sample(
    doc_stats: list[dict], num_docs: int, seed: int
) -> list[dict]:
    """Sample docs stratified by boundary density (low/medium/high)."""
    rng = random.Random(seed)

    sorted_stats = sorted(doc_stats, key=lambda x: x["density"])
    n = len(sorted_stats)
    tercile = n // 3

    low = sorted_stats[:tercile]
    mid = sorted_stats[tercile : 2 * tercile]
    high = sorted_stats[2 * tercile :]

    per_bucket = num_docs // 3
    remainder = num_docs - 3 * per_bucket

    sampled = []
    for bucket in [low, mid, high]:
        k = min(per_bucket, len(bucket))
        sampled.extend(rng.sample(bucket, k))

    if remainder > 0 and len(doc_stats) > len(sampled):
        remaining = [s for s in doc_stats if s not in sampled]
        extra = min(remainder, len(remaining))
        sampled.extend(rng.sample(remaining, extra))

    return sampled


def main():
    parser = argparse.ArgumentParser(description="Sample benchmark dataset")
    parser.add_argument("--num-docs", type=int, default=None, help="Number of docs to sample (default: all test docs)")
    parser.add_argument("--max-windows", type=int, default=None, help="Cap total windows in benchmark")
    parser.add_argument("--output-dir", type=Path, default=BENCHMARK_DIR)
    args = parser.parse_args()

    print("Loading annotations and test split info...")
    with open(ANNOTATIONS_FILE) as f:
        annotations = json.load(f)
    test_ids = load_test_doc_ids()
    print(f"  Test documents (1 per work): {len(test_ids)}")

    doc_stats = load_doc_stats(test_ids, annotations)
    print(f"  With text files: {len(doc_stats)}")

    work_ids_in_benchmark = set(s["work_id"] for s in doc_stats)
    print(f"  Unique works: {len(work_ids_in_benchmark)}")

    if args.num_docs and args.num_docs < len(doc_stats):
        sampled = stratified_sample(doc_stats, args.num_docs, SEED)
        print(f"  Sampled {len(sampled)} docs (stratified by density)")
    else:
        sampled = doc_stats
        print(f"  Using all {len(sampled)} test docs")

    total_bp = sum(s["num_breakpoints"] for s in sampled)
    print(f"  Total breakpoints in benchmark: {total_bp}")

    dataset_path = PROCESSED_DIR / "dataset"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_path} not found. Run prepare_data.py first."
        )
    ds = load_from_disk(str(dataset_path))
    test_ds = ds["test"]

    sampled_ids = {s["doc_id"] for s in sampled}
    benchmark_ds = test_ds.filter(lambda x: x["doc_id"] in sampled_ids)

    if args.max_windows and len(benchmark_ds) > args.max_windows:
        indices = list(range(len(benchmark_ds)))
        random.Random(SEED).shuffle(indices)
        benchmark_ds = benchmark_ds.select(indices[: args.max_windows])

    print(f"  Benchmark windows: {len(benchmark_ds)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_ds.save_to_disk(str(args.output_dir / "dataset"))

    benchmark_meta = {
        "num_docs": len(sampled),
        "num_unique_works": len(set(s["work_id"] for s in sampled)),
        "num_windows": len(benchmark_ds),
        "total_breakpoints": total_bp,
        "docs": sampled,
    }
    with open(args.output_dir / "benchmark_meta.json", "w") as f:
        json.dump(benchmark_meta, f, indent=2)

    print(f"\nBenchmark saved to {args.output_dir}")

    print("\nBenchmark statistics:")
    densities = [s["density"] for s in sampled]
    print(f"  Density (bp/10K chars): min={min(densities):.2f}, "
          f"median={sorted(densities)[len(densities)//2]:.2f}, "
          f"max={max(densities):.2f}")
    lengths = [s["text_length"] for s in sampled]
    print(f"  Doc length: min={min(lengths)//1024}KB, "
          f"median={sorted(lengths)[len(lengths)//2]//1024}KB, "
          f"max={max(lengths)//1024}KB")


if __name__ == "__main__":
    main()
