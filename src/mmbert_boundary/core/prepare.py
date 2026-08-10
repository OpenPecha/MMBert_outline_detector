"""Prepare training data for mmBERT boundary detection.

Input: one JSON file per volume with keys:
  - filename  : volume identifier string
  - content   : full volume text (UTF-8)
  - segments  : list of {span_start, span_end, label}

Breakpoints are derived as ``span_start`` of every segment after the first.
Documents are deduplicated by BDRC work ID so no volumes of the same work
leak across train/val/test splits. Tokenization is parallelised across CPU
workers with ProcessPoolExecutor.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import unicodedata
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from tqdm import tqdm
from transformers import AutoTokenizer

from mmbert_boundary.config import (
    ANNOTATED_DATA_DIR,
    BOUNDARY_RADIUS,
    LABEL_B,
    LABEL_O,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    NEG_SAMPLE_RATIO,
    NUM_WORKERS,
    PROCESSED_DIR,
    SEED,
    STRIDE,
    TEST_RATIO,
    VAL_RATIO,
)

ARROW_FEATURES = Features({
    "input_ids": Sequence(Value("int32")),
    "attention_mask": Sequence(Value("int32")),
    "labels": Sequence(Value("int32")),
    "doc_id": Value("string"),
    "has_boundary": Value("bool"),
})

_KEEP_KEYS = ("input_ids", "attention_mask", "labels", "doc_id", "has_boundary")

_ARROW_SCHEMA = pa.schema([
    ("input_ids", pa.list_(pa.int32())),
    ("attention_mask", pa.list_(pa.int32())),
    ("labels", pa.list_(pa.int32())),
    ("doc_id", pa.string()),
    ("has_boundary", pa.bool_()),
])


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VolumeMeta:
    """Lightweight metadata extracted from a volume JSON.

    Only stores fields needed for grouping and splitting — the full text and
    segments are left on disk and loaded by workers on demand.

    Attributes:
        doc_id: Stem of the source JSON file (== filename field).
        work_id: BDRC work identifier (first underscore-separated segment).
        num_breakpoints: Count of segment transitions.
    """

    doc_id: str
    work_id: str
    num_breakpoints: int


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def scan_volume_meta(path: Path) -> VolumeMeta:
    """Read a volume JSON and extract only grouping/splitting metadata.

    Args:
        path: Path to a volume JSON file.

    Returns:
        Lightweight VolumeMeta.

    Raises:
        ValueError: If the JSON is malformed or missing required keys.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise ValueError(f"Cannot load {path}: {err}") from err

    filename: str = payload.get("filename") or path.stem
    segments: list[dict[str, Any]] = payload.get("segments", [])
    num_bp = len(
        {seg["span_start"] for seg in segments[1:] if seg.get("span_start") is not None}
    )
    work_id = filename.split("_", 1)[0]
    return VolumeMeta(doc_id=filename, work_id=work_id, num_breakpoints=num_bp)


def load_volume_for_tokenization(path: Path) -> tuple[str, list[int]]:
    """Load a volume's text and breakpoints for tokenization.

    Called inside worker subprocesses — only the fields needed for
    tokenization are returned.

    Args:
        path: Path to a volume JSON file.

    Returns:
        Tuple of ``(nfc_normalised_text, sorted_breakpoints)``.

    Raises:
        ValueError: If the JSON is malformed or missing required keys.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise ValueError(f"Cannot load {path}: {err}") from err

    raw_text: str = payload.get("content", "")
    text = unicodedata.normalize("NFC", raw_text)
    segments: list[dict[str, Any]] = payload.get("segments", [])
    breakpoints = sorted(
        {seg["span_start"] for seg in segments[1:] if seg.get("span_start") is not None}
    )
    return text, breakpoints


def scan_all_volume_metas(data_dir: Path) -> list[VolumeMeta]:
    """Scan metadata for every volume JSON in a directory.

    Args:
        data_dir: Directory containing ``*.json`` volume files.

    Returns:
        List of VolumeMeta in deterministic (sorted filename) order.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    paths = sorted(data_dir.glob("*.json"))
    metas: list[VolumeMeta] = []
    errors = 0
    for path in paths:
        try:
            metas.append(scan_volume_meta(path))
        except ValueError as err:
            print(f"[WARN] Skipping {path.name}: {err}")
            errors += 1

    if errors:
        print(f"[WARN] {errors} files could not be loaded and were skipped.")

    return metas


# ---------------------------------------------------------------------------
# Work-ID grouping and splitting
# ---------------------------------------------------------------------------

def group_by_work(metas: list[VolumeMeta]) -> dict[str, list[VolumeMeta]]:
    """Group volume metadata by BDRC work ID.

    Args:
        metas: List of lightweight VolumeMeta objects.

    Returns:
        Mapping from work_id to list of VolumeMeta.
    """
    groups: dict[str, list[VolumeMeta]] = defaultdict(list)
    for meta in metas:
        groups[meta.work_id].append(meta)
    return dict(groups)


def split_works(
    work_ids: list[str],
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Split work IDs into train/val/test so no work appears in multiple splits.

    Args:
        work_ids: All unique work IDs.
        test_ratio: Fraction for test.
        val_ratio: Fraction for validation.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of ``(train_wids, val_wids, test_wids)``.
    """
    rng = random.Random(seed)
    shuffled = list(work_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))

    test_wids = shuffled[:n_test]
    val_wids = shuffled[n_test : n_test + n_val]
    train_wids = shuffled[n_test + n_val :]

    return train_wids, val_wids, test_wids


# ---------------------------------------------------------------------------
# Tokenization and labeling
# ---------------------------------------------------------------------------

def create_boundary_set(breakpoints: list[int], radius: int = 0) -> set[int]:
    """Expand each breakpoint into a character-offset band.

    Args:
        breakpoints: Exact boundary character offsets.
        radius: Number of extra chars on each side to mark as boundary.

    Returns:
        Set of character offsets that should receive the B label.
    """
    boundary_chars: set[int] = set()
    for bp in breakpoints:
        for offset in range(-radius, radius + 1):
            boundary_chars.add(bp + offset)
    return boundary_chars


def tokenize_and_label(
    text: str,
    breakpoints: list[int],
    tokenizer: Any,
    max_length: int,
    stride: int,
    boundary_radius: int,
) -> list[dict[str, Any]]:
    """Tokenize a full document with sliding windows and assign B/O labels.

    Args:
        text: Full volume text.
        breakpoints: Derived boundary character offsets.
        tokenizer: HuggingFace fast tokenizer.
        max_length: Sliding-window size in tokens.
        stride: Token overlap between consecutive windows.
        boundary_radius: Character radius around each breakpoint to label B.

    Returns:
        List of example dicts with ``input_ids``, ``attention_mask``,
        ``labels``, ``has_boundary``, and ``boundary_char_positions``.
    """
    boundary_chars = create_boundary_set(breakpoints, boundary_radius)

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        max_length=max_length,
        stride=stride,
        truncation=True,
        padding="max_length",
    )

    examples: list[dict[str, Any]] = []
    for window_idx in range(len(encoding["input_ids"])):
        input_ids = encoding["input_ids"][window_idx]
        attention_mask = encoding["attention_mask"][window_idx]
        offsets = encoding["offset_mapping"][window_idx]

        labels: list[int] = []
        window_boundary_positions: list[int] = []

        for start, end in offsets:
            if start == 0 and end == 0:
                labels.append(-100)
                continue

            token_has_boundary = any(c in boundary_chars for c in range(start, end))

            if token_has_boundary:
                labels.append(LABEL_B)
                window_boundary_positions.append(start)
            else:
                labels.append(LABEL_O)

        examples.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "has_boundary": bool(window_boundary_positions),
                "boundary_char_positions": window_boundary_positions,
            }
        )

    return examples


# ---------------------------------------------------------------------------
# Parallel worker (module-level so it is picklable)
# ---------------------------------------------------------------------------

_worker_tokenizer: Any = None


def _worker_init(tokenizer_name: str) -> None:
    """Per-process initializer: load the tokenizer once and cache it.

    Args:
        tokenizer_name: HuggingFace model/tokenizer identifier.
    """
    global _worker_tokenizer  # noqa: PLW0603
    _worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


def _worker_process_volume(
    args: tuple[str, str, int, int, int],
) -> tuple[str, list[dict[str, Any]], str]:
    """Worker function executed in a subprocess.

    Args:
        args: Tuple of (json_path_str, doc_id, max_length, stride,
              boundary_radius).

    Returns:
        Tuple of (doc_id, examples, error_message). error_message is
        empty on success.
    """
    json_path_str, doc_id, max_length, stride, boundary_radius = args
    try:
        text, breakpoints = load_volume_for_tokenization(Path(json_path_str))
        examples = tokenize_and_label(
            text,
            breakpoints,
            _worker_tokenizer,
            max_length,
            stride,
            boundary_radius,
        )
        for ex in examples:
            ex["doc_id"] = doc_id
        return doc_id, examples, ""
    except Exception as err:  # noqa: BLE001
        return doc_id, [], str(err)


# ---------------------------------------------------------------------------
# Parallel and sequential processing orchestrators
# ---------------------------------------------------------------------------

def _build_worker_args(
    doc_ids: list[str],
    data_dir: Path,
    max_length: int,
    stride: int,
    boundary_radius: int,
) -> list[tuple[str, str, int, int, int]]:
    """Build the argument tuples passed to each worker.

    Args:
        doc_ids: Document identifiers (JSON filenames without extension).
        data_dir: Directory that contains the JSON files.
        max_length: Sliding-window token length.
        stride: Token overlap.
        boundary_radius: Char radius for boundary labeling.

    Returns:
        List of argument tuples ready for ``_worker_process_volume``.
    """
    return [
        (str(data_dir / f"{doc_id}.json"), doc_id, max_length, stride, boundary_radius)
        for doc_id in doc_ids
    ]


_MAX_INFLIGHT = 32


def process_docs_to_arrow(
    doc_ids: list[str],
    data_dir: Path,
    tokenizer_name: str,
    max_length: int,
    stride: int,
    boundary_radius: int,
    num_workers: int,
    arrow_path: Path,
    desc: str = "",
) -> Dataset:
    """Tokenize volumes and stream results to an Arrow file on disk.

    Instead of accumulating every window in a Python list (which OOMs on
    large corpora), each worker's output is written to an Arrow file as
    soon as it arrives.  The returned Dataset is memory-mapped from that
    file, so RAM usage stays bounded.

    Args:
        doc_ids: Document identifiers (JSON filenames without extension).
        data_dir: Directory containing the JSON source files.
        tokenizer_name: HuggingFace model name.
        max_length: Sliding-window token length.
        stride: Token overlap.
        boundary_radius: Char radius for boundary labeling.
        num_workers: Number of parallel workers. 1 = sequential.
        arrow_path: Destination ``.arrow`` file.
        desc: Progress bar label.

    Returns:
        Memory-mapped HuggingFace Dataset backed by *arrow_path*.
    """
    worker_args = _build_worker_args(doc_ids, data_dir, max_length, stride, boundary_radius)

    arrow_path.parent.mkdir(parents=True, exist_ok=True)
    sink = pa.OSFile(str(arrow_path), "wb")
    writer = pa.ipc.new_stream(sink, _ARROW_SCHEMA)
    failed: list[str] = []
    written = 0

    def _flush_examples(examples: list[dict[str, Any]]) -> int:
        if not examples:
            return 0
        batch = pa.record_batch(
            {k: [ex[k] for ex in examples] for k in _KEEP_KEYS},
            schema=_ARROW_SCHEMA,
        )
        writer.write_batch(batch)
        return len(examples)

    if num_workers == 1:
        _worker_init(tokenizer_name)
        for arg_tuple in tqdm(worker_args, desc=desc, unit="vol"):
            doc_id, examples, error = _worker_process_volume(arg_tuple)
            if error:
                print(f"\n[ERROR] {doc_id}: {error}")
                failed.append(doc_id)
            else:
                written += _flush_examples(examples)
            del examples
    else:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_worker_init,
            initargs=(tokenizer_name,),
        ) as executor:
            it = iter(worker_args)
            pending: dict[Any, str] = {}
            done_count = 0

            def _drain_one() -> None:
                nonlocal written, done_count
                finished = next(as_completed(pending))
                doc_id_done = pending.pop(finished)
                _, examples, error = finished.result()
                if error:
                    pbar.write(f"[ERROR] {doc_id_done}: {error}")
                    failed.append(doc_id_done)
                else:
                    written += _flush_examples(examples)
                del examples, finished
                done_count += 1
                pbar.update(1)
                if done_count % 100 == 0:
                    gc.collect()

            with tqdm(total=len(worker_args), desc=desc, unit="vol") as pbar:
                for arg in it:
                    if len(pending) >= _MAX_INFLIGHT:
                        _drain_one()
                    fut = executor.submit(_worker_process_volume, arg)
                    pending[fut] = arg[1]

                while pending:
                    _drain_one()

    if failed:
        print(f"\n[WARN] {len(failed)} volumes failed during processing: {failed}")

    writer.close()
    sink.close()
    gc.collect()
    print(f"  {desc}: {written} windows → {arrow_path.name}")
    return Dataset.from_file(str(arrow_path))


# ---------------------------------------------------------------------------
# Negative downsampling
# ---------------------------------------------------------------------------

def count_labels(dataset: Dataset, batch_size: int = 1000) -> tuple[int, int]:
    """Count B and O labels without full materialisation.

    Args:
        dataset: HuggingFace Dataset with a ``labels`` column.
        batch_size: Rows loaded per batch to keep memory bounded.

    Returns:
        Tuple of ``(b_count, o_count)``.
    """
    b_count = 0
    o_count = 0
    for start in range(0, len(dataset), batch_size):
        batch_labels = dataset[start : start + batch_size]["labels"]
        for labels in batch_labels:
            for lbl in labels:
                if lbl == LABEL_B:
                    b_count += 1
                elif lbl == LABEL_O:
                    o_count += 1
    return b_count, o_count


def downsample_negatives(
    dataset: Dataset,
    neg_ratio: float,
    seed: int,
) -> Dataset:
    """Keep all windows with >= 1 B label; subsample O-only windows.

    Relies on the precomputed ``has_boundary`` bool column written during
    tokenization, so this is an index filter — no per-token label scan.

    Args:
        dataset: Disk-backed HuggingFace Dataset with ``has_boundary``.
        neg_ratio: Fraction of O-only windows to retain (0.0–1.0).
        seed: RNG seed for reproducibility.

    Returns:
        Filtered Dataset (new Arrow table, not a view).

    Raises:
        KeyError: If ``has_boundary`` is missing from *dataset*.
    """
    if "has_boundary" not in dataset.column_names:
        raise KeyError(
            "Dataset is missing 'has_boundary'; re-run prepare-data so the "
            "column is written during tokenization."
        )

    rng = random.Random(seed)
    flags = dataset["has_boundary"]
    pos_indices = [i for i, flag in enumerate(flags) if flag]
    neg_indices = [i for i, flag in enumerate(flags) if not flag]

    n_keep = max(1, int(len(neg_indices) * neg_ratio)) if neg_indices else 0
    sampled_neg = rng.sample(neg_indices, min(n_keep, len(neg_indices)))

    keep_indices = pos_indices + sampled_neg
    rng.shuffle(keep_indices)

    print(
        f"    Kept {len(pos_indices)} B-windows + {len(sampled_neg)}/{len(neg_indices)} "
        f"O-only windows = {len(keep_indices)} total"
    )
    return dataset.select(keep_indices)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Run corpus preprocessing CLI.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(description="Prepare mmBERT training data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ANNOTATED_DATA_DIR,
        help="Directory containing per-volume JSON files (default: %(default)s)",
    )
    parser.add_argument(
        "--max-works",
        type=int,
        default=None,
        help="Limit number of unique works processed (useful for dry runs)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help="Sliding-window size in tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=STRIDE,
        help="Token overlap between consecutive windows (default: %(default)s)",
    )
    parser.add_argument(
        "--boundary-radius",
        type=int,
        default=BOUNDARY_RADIUS,
        help="Char radius around each breakpoint to label B (default: %(default)s)",
    )
    parser.add_argument(
        "--neg-sample-ratio",
        type=float,
        default=NEG_SAMPLE_RATIO,
        help="Fraction of O-only windows to keep in training set (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory for output dataset and metadata (default: %(default)s)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help="HuggingFace tokenizer model ID (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=NUM_WORKERS or os.cpu_count() or 1,
        help="Number of parallel worker processes (default: CPU count)",
    )
    args = parser.parse_args(argv)

    print(f"Data directory:  {args.data_dir}")
    print(f"Model/tokenizer: {args.model_name}")
    print(f"Workers:         {args.workers}")

    print("\nScanning volume metadata...")
    all_metas = scan_all_volume_metas(args.data_dir)
    print(f"  Found {len(all_metas)} volumes")

    print("\nGrouping by BDRC work ID...")
    work_groups = group_by_work(all_metas)
    total_bp = sum(m.num_breakpoints for m in all_metas)
    print(f"  Unique works:  {len(work_groups)}")
    print(f"  Total breakpoints derived from segments: {total_bp}")

    work_ids = list(work_groups.keys())
    if args.max_works:
        work_ids = work_ids[: args.max_works]
        print(f"  Limited to {len(work_ids)} works (--max-works)")

    deduped: dict[str, VolumeMeta] = {wid: work_groups[wid][0] for wid in work_ids}
    extra_volumes = sum(len(work_groups[wid]) - 1 for wid in work_ids)
    print(f"  Representative docs (1 per work): {len(deduped)}")
    print(f"  Extra volumes available for training: {extra_volumes}")

    print("\nSplitting by work ID...")
    train_wids, val_wids, test_wids = split_works(work_ids, TEST_RATIO, VAL_RATIO, SEED)
    print(f"  Train works: {len(train_wids)}")
    print(f"  Val works:   {len(val_wids)}")
    print(f"  Test works:  {len(test_wids)}")

    train_doc_ids: list[str] = []
    for wid in train_wids:
        train_doc_ids.extend(m.doc_id for m in work_groups[wid])
    val_doc_ids = [deduped[wid].doc_id for wid in val_wids]
    test_doc_ids = [deduped[wid].doc_id for wid in test_wids]

    print(f"\n  Train docs (all volumes): {len(train_doc_ids)}")
    print(f"  Val docs (1 per work):    {len(val_doc_ids)}")
    print(f"  Test docs (1 per work):   {len(test_doc_ids)}")

    del all_metas, work_groups, deduped

    print(f"\nTokenizing with max_length={args.max_length}, stride={args.stride}, "
          f"workers={args.workers}...")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.output_dir / "_tmp_arrows"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    train_ds = process_docs_to_arrow(
        train_doc_ids, args.data_dir, args.model_name,
        args.max_length, args.stride, args.boundary_radius,
        args.workers, tmp_dir / "train_raw.arrow", desc="Train",
    )
    val_ds = process_docs_to_arrow(
        val_doc_ids, args.data_dir, args.model_name,
        args.max_length, args.stride, args.boundary_radius,
        args.workers, tmp_dir / "val.arrow", desc="Val",
    )
    test_ds = process_docs_to_arrow(
        test_doc_ids, args.data_dir, args.model_name,
        args.max_length, args.stride, args.boundary_radius,
        args.workers, tmp_dir / "test.arrow", desc="Test",
    )

    print("\nScanning label distribution...")
    raw_train_b, raw_train_o = count_labels(train_ds)
    print(f"\n  Raw train windows: {len(train_ds)}")
    print(f"  Raw label distribution: B={raw_train_b}, O={raw_train_o}, "
          f"ratio=1:{raw_train_o // max(raw_train_b, 1)}")

    print(f"\n  Downsampling O-only windows (keep {args.neg_sample_ratio:.0%})...")
    train_ds = downsample_negatives(train_ds, args.neg_sample_ratio, SEED)

    train_b, train_o = count_labels(train_ds)
    print(f"\n  Final train windows: {len(train_ds)}")
    print(f"  Val windows:         {len(val_ds)}")
    print(f"  Test windows:        {len(test_ds)}")
    print(f"  Final label dist:    B={train_b}, O={train_o}, "
          f"ratio=1:{train_o // max(train_b, 1)}")

    print("\nSaving datasets...")
    ds = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})
    dataset_path = args.output_dir / "dataset"
    ds.save_to_disk(str(dataset_path))
    print(f"  Saved dataset to {dataset_path}")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    split_info = {
        "train_work_ids": sorted(train_wids),
        "val_work_ids": sorted(val_wids),
        "test_work_ids": sorted(test_wids),
        "train_doc_ids": train_doc_ids,
        "val_doc_ids": val_doc_ids,
        "test_doc_ids": test_doc_ids,
    }
    with open(args.output_dir / "split_info.json", "w", encoding="utf-8") as fh:
        json.dump(split_info, fh, indent=2)

    with open(args.output_dir / "data_config.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model_name": args.model_name,
                "max_length": args.max_length,
                "stride": args.stride,
                "boundary_radius": args.boundary_radius,
                "neg_sample_ratio": args.neg_sample_ratio,
                "num_workers": args.workers,
                "num_train_works": len(train_wids),
                "num_val_works": len(val_wids),
                "num_test_works": len(test_wids),
                "num_train_docs": len(train_doc_ids),
                "num_val_docs": len(val_doc_ids),
                "num_test_docs": len(test_doc_ids),
                "raw_train_windows": raw_train_b + raw_train_o,
                "num_train_windows": len(train_ds),
                "num_val_windows": len(val_ds),
                "num_test_windows": len(test_ds),
                "label_b_count": train_b,
                "label_o_count": train_o,
                "raw_label_b_count": raw_train_b,
                "raw_label_o_count": raw_train_o,
            },
            fh,
            indent=2,
        )

    print("\nDone!")
