"""Sample a stratified benchmark dataset for consistent evaluation.

Reads per-volume JSON files from the annotated data directory (each with
``filename``, ``content``, and ``segments`` keys) to compute boundary
density, then stratifies by density for a compact benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
import unicodedata
from pathlib import Path

from datasets import load_from_disk

from mmbert_boundary.config import (
    ANNOTATED_DATA_DIR,
    BENCHMARK_DIR,
    PROCESSED_DIR,
    SEED,
)


def load_test_doc_ids() -> list[str]:
    """Read the test document IDs persisted by prepare_data.

    Returns:
        List of doc_id strings for the test split.

    Raises:
        FileNotFoundError: If ``split_info.json`` does not exist yet.
    """
    split_info_path = PROCESSED_DIR / "split_info.json"
    if not split_info_path.exists():
        raise FileNotFoundError(
            f"{split_info_path} not found. Run `mmbert-boundary prepare-data` first."
        )
    with open(split_info_path) as f:
        split_info = json.load(f)
    return split_info["test_doc_ids"]


def load_doc_stats(doc_ids: list[str], data_dir: Path) -> list[dict]:
    """Compute per-document statistics from source JSON files.

    Args:
        doc_ids: Document identifiers (JSON filenames without extension).
        data_dir: Directory containing ``{doc_id}.json`` files.

    Returns:
        List of stat dicts with ``doc_id``, ``work_id``, ``text_length``,
        ``num_breakpoints``, and ``density`` (breakpoints per 10 K chars).
    """
    stats: list[dict] = []
    skipped = 0
    for doc_id in doc_ids:
        json_path = data_dir / f"{doc_id}.json"
        if not json_path.exists():
            skipped += 1
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped += 1
            continue

        filename: str = payload.get("filename") or doc_id
        content: str = payload.get("content", "")
        text_len = len(unicodedata.normalize("NFC", content))

        segments = payload.get("segments", [])
        breakpoints = {
            seg["span_start"]
            for seg in segments[1:]
            if seg.get("span_start") is not None
        }
        n_bp = len(breakpoints)
        density = n_bp / max(text_len, 1) * 10000

        stats.append({
            "doc_id": doc_id,
            "work_id": filename.split("_", 1)[0],
            "filename": filename,
            "text_length": text_len,
            "num_breakpoints": n_bp,
            "density": density,
        })

    if skipped:
        print(f"  [WARN] {skipped} test docs not found in {data_dir}")
    return stats


def stratified_sample(doc_stats: list[dict], num_docs: int, seed: int) -> list[dict]:
    """Sample docs stratified by boundary density into low/medium/high buckets.

    Args:
        doc_stats: List of per-document stat dicts.
        num_docs: Total number of docs to sample.
        seed: RNG seed for reproducibility.

    Returns:
        Sampled list of doc stat dicts.
    """
    rng = random.Random(seed)

    sorted_stats = sorted(doc_stats, key=lambda x: x["density"])
    n = len(sorted_stats)
    tercile = n // 3

    low = sorted_stats[:tercile]
    mid = sorted_stats[tercile : 2 * tercile]
    high = sorted_stats[2 * tercile :]

    per_bucket = num_docs // 3
    remainder = num_docs - 3 * per_bucket

    sampled: list[dict] = []
    for bucket in [low, mid, high]:
        k = min(per_bucket, len(bucket))
        sampled.extend(rng.sample(bucket, k))

    if remainder > 0 and len(doc_stats) > len(sampled):
        remaining = [s for s in doc_stats if s not in sampled]
        extra = min(remainder, len(remaining))
        sampled.extend(rng.sample(remaining, extra))

    return sampled


def main(argv: list[str] | None = None) -> None:
    """Sample a stratified benchmark dataset from the test split.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(description="Sample benchmark dataset")
    parser.add_argument(
        "--num-docs",
        type=int,
        default=None,
        help="Number of docs to sample (default: all test docs)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Cap total windows in benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCHMARK_DIR,
        help="Destination directory for benchmark dataset (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    print("Loading test split info and scanning source JSONs...")
    test_ids = load_test_doc_ids()
    print(f"  Test documents (1 per work): {len(test_ids)}")

    doc_stats = load_doc_stats(test_ids, ANNOTATED_DATA_DIR)
    print(f"  With source JSONs: {len(doc_stats)}")

    work_ids_in_benchmark = {s["work_id"] for s in doc_stats}
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
            f"{dataset_path} not found. Run `mmbert-boundary prepare-data` first."
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
        "num_unique_works": len({s["work_id"] for s in sampled}),
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
