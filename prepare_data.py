"""
Prepare training data for mmBERT boundary detection.

Deduplicates by BDRC work ID (first segment of filename) so that volumes
of the same collected work never leak across train/val/test splits.
Test and val splits use one document per work; training uses all volumes.

Usage:
    python prepare_data.py
    python prepare_data.py --max-works 50   # quick test run
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer

from config import (
    ANNOTATIONS_FILE,
    BOUNDARY_RADIUS,
    DOCUMENTS_DIR,
    LABEL_B,
    LABEL_O,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    NEG_SAMPLE_RATIO,
    PROCESSED_DIR,
    SEED,
    STRIDE,
    TEST_RATIO,
    VAL_RATIO,
)


def load_annotations():
    with open(ANNOTATIONS_FILE) as f:
        return json.load(f)


def main_work_id(filename: str) -> str:
    """Return main work ID from filename, e.g. W8LS73547 from W8LS73547_I8LS73558_..."""
    return filename.split("_", 1)[0]


def get_available_docs(annotations: dict) -> list[tuple[str, dict]]:
    available = []
    for doc_id, ann in annotations.items():
        doc_path = DOCUMENTS_DIR / f"{doc_id}.txt"
        if doc_path.exists():
            available.append((doc_id, ann))
    return available


def group_by_work(docs: list[tuple[str, dict]]) -> dict[str, list[tuple[str, dict]]]:
    """Group documents by their main BDRC work ID."""
    groups = defaultdict(list)
    for doc_id, ann in docs:
        wid = main_work_id(ann["filename"])
        groups[wid].append((doc_id, ann))
    return dict(groups)


def deduplicate_work_group(
    work_docs: list[tuple[str, dict]],
) -> tuple[str, dict]:
    """Pick one representative document per work (first in list)."""
    return work_docs[0]


def create_boundary_set(breakpoints: list[int], radius: int = 0) -> set[int]:
    boundary_chars = set()
    for bp in breakpoints:
        for offset in range(-radius, radius + 1):
            boundary_chars.add(bp + offset)
    return boundary_chars


def tokenize_and_label(
    text: str,
    breakpoints: list[int],
    tokenizer,
    max_length: int,
    stride: int,
    boundary_radius: int,
) -> list[dict]:
    """
    Tokenize a full document with sliding windows and assign B/O labels.

    Returns a list of examples, each with:
      - input_ids, attention_mask
      - labels (list of 0/1, -100 for special tokens and overlap)
      - boundary_char_positions (for evaluation)
    """
    boundary_chars = create_boundary_set(breakpoints, boundary_radius)

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        max_length=max_length,
        stride=stride,
        truncation=True,
        padding=False,
    )

    examples = []
    for window_idx in range(len(encoding["input_ids"])):
        input_ids = encoding["input_ids"][window_idx]
        attention_mask = encoding["attention_mask"][window_idx]
        offsets = encoding["offset_mapping"][window_idx]

        labels = []
        window_boundary_positions = []

        for token_idx, (start, end) in enumerate(offsets):
            if start == 0 and end == 0:
                labels.append(-100)
                continue

            token_has_boundary = any(
                c in boundary_chars for c in range(start, end)
            )

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
                "boundary_char_positions": window_boundary_positions,
            }
        )

    return examples


def split_works(
    work_ids: list[str],
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Split work IDs into train/val/test so no work appears in multiple splits."""
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


def process_docs(
    docs: list[tuple[str, dict]],
    tokenizer,
    max_length: int,
    stride: int,
    boundary_radius: int,
    desc: str = "",
) -> list[dict]:
    all_examples = []
    for doc_id, ann in tqdm(docs, desc=desc):
        doc_path = DOCUMENTS_DIR / f"{doc_id}.txt"
        text = doc_path.read_text(encoding="utf-8")
        breakpoints = ann["breakpoints"]

        examples = tokenize_and_label(
            text, breakpoints, tokenizer, max_length, stride, boundary_radius
        )
        for ex in examples:
            ex["doc_id"] = doc_id
        all_examples.extend(examples)

    return all_examples


def downsample_negatives(
    examples: list[dict],
    neg_ratio: float,
    seed: int,
) -> list[dict]:
    """Keep all windows with at least one B label; subsample O-only windows.

    Args:
        examples: Token-classified sliding windows.
        neg_ratio: Fraction of O-only windows to retain (0.0–1.0).
        seed: RNG seed for reproducibility.

    Returns:
        Filtered list of examples.
    """
    rng = random.Random(seed)
    positives = []
    negatives = []

    for ex in examples:
        has_boundary = any(l == LABEL_B for l in ex["labels"])
        if has_boundary:
            positives.append(ex)
        else:
            negatives.append(ex)

    n_keep = max(1, int(len(negatives) * neg_ratio))
    sampled_negatives = rng.sample(negatives, n_keep)

    result = positives + sampled_negatives
    rng.shuffle(result)

    print(f"    Kept {len(positives)} B-windows + {n_keep}/{len(negatives)} O-only windows "
          f"= {len(result)} total")
    return result


def examples_to_dataset(examples: list[dict]) -> Dataset:
    if not examples:
        return Dataset.from_dict(
            {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
                "doc_id": [],
            }
        )

    return Dataset.from_dict(
        {
            "input_ids": [ex["input_ids"] for ex in examples],
            "attention_mask": [ex["attention_mask"] for ex in examples],
            "labels": [ex["labels"] for ex in examples],
            "doc_id": [ex["doc_id"] for ex in examples],
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare mmBERT training data")
    parser.add_argument("--max-works", type=int, default=None, help="Limit number of unique works (for testing)")
    parser.add_argument("--max-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--boundary-radius", type=int, default=BOUNDARY_RADIUS)
    parser.add_argument("--neg-sample-ratio", type=float, default=NEG_SAMPLE_RATIO,
                        help="Fraction of O-only windows to keep in training set (0.0–1.0)")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Loading annotations...")
    annotations = load_annotations()
    docs = get_available_docs(annotations)
    print(f"  Found {len(docs)} documents with text files")

    print("\nGrouping by BDRC work ID...")
    work_groups = group_by_work(docs)
    print(f"  Unique works: {len(work_groups)}")
    total_bp = sum(len(ann["breakpoints"]) for _, ann in docs)
    print(f"  Total breakpoints: {total_bp}")

    # Deduplicate: one representative doc per work
    deduped = {}
    skipped_count = 0
    for wid, work_docs in work_groups.items():
        deduped[wid] = deduplicate_work_group(work_docs)
        skipped_count += len(work_docs) - 1
    print(f"  Representative docs (1 per work): {len(deduped)}")
    print(f"  Extra volumes (available for training): {skipped_count}")

    work_ids = list(work_groups.keys())
    if args.max_works:
        work_ids = work_ids[: args.max_works]
        print(f"  Limited to {len(work_ids)} works")

    # Split at work level
    print("\nSplitting by work ID...")
    train_wids, val_wids, test_wids = split_works(
        work_ids, TEST_RATIO, VAL_RATIO, SEED
    )
    print(f"  Train works: {len(train_wids)}")
    print(f"  Val works:   {len(val_wids)}")
    print(f"  Test works:  {len(test_wids)}")

    # Train: use ALL volumes for train works (more data diversity)
    train_docs = []
    for wid in train_wids:
        train_docs.extend(work_groups[wid])

    # Val/Test: use only the deduplicated representative (one per work)
    val_docs = [deduped[wid] for wid in val_wids]
    test_docs = [deduped[wid] for wid in test_wids]

    print(f"\n  Train docs (all volumes): {len(train_docs)}")
    print(f"  Val docs (1 per work):    {len(val_docs)}")
    print(f"  Test docs (1 per work):   {len(test_docs)}")

    # Save split info
    split_info = {
        "train_work_ids": sorted(train_wids),
        "val_work_ids": sorted(val_wids),
        "test_work_ids": sorted(test_wids),
        "train_doc_ids": [doc_id for doc_id, _ in train_docs],
        "val_doc_ids": [doc_id for doc_id, _ in val_docs],
        "test_doc_ids": [doc_id for doc_id, _ in test_docs],
    }

    print(f"\nTokenizing with max_length={args.max_length}, stride={args.stride}...")
    train_examples = process_docs(
        train_docs, tokenizer, args.max_length, args.stride, args.boundary_radius, "Train"
    )
    val_examples = process_docs(
        val_docs, tokenizer, args.max_length, args.stride, args.boundary_radius, "Val"
    )
    test_examples = process_docs(
        test_docs, tokenizer, args.max_length, args.stride, args.boundary_radius, "Test"
    )

    raw_train_b = sum(1 for ex in train_examples for l in ex["labels"] if l == LABEL_B)
    raw_train_o = sum(1 for ex in train_examples for l in ex["labels"] if l == LABEL_O)
    print(f"\n  Raw train windows: {len(train_examples)}")
    print(f"  Raw label distribution: B={raw_train_b}, O={raw_train_o}, ratio=1:{raw_train_o // max(raw_train_b, 1)}")

    print(f"\n  Downsampling O-only windows (keep {args.neg_sample_ratio:.0%})...")
    train_examples = downsample_negatives(train_examples, args.neg_sample_ratio, SEED)

    train_b = sum(1 for ex in train_examples for l in ex["labels"] if l == LABEL_B)
    train_o = sum(1 for ex in train_examples for l in ex["labels"] if l == LABEL_O)
    print(f"\n  Final train windows: {len(train_examples)}")
    print(f"  Val windows:         {len(val_examples)}")
    print(f"  Test windows:        {len(test_examples)}")
    print(f"  Final label distribution: B={train_b}, O={train_o}, ratio=1:{train_o // max(train_b, 1)}")

    print("\nSaving datasets...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ds = DatasetDict(
        {
            "train": examples_to_dataset(train_examples),
            "validation": examples_to_dataset(val_examples),
            "test": examples_to_dataset(test_examples),
        }
    )
    ds.save_to_disk(str(args.output_dir / "dataset"))
    print(f"  Saved to {args.output_dir / 'dataset'}")

    with open(args.output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # Save dedupe report (mirrors the benchmark script format)
    dedupe_report = {
        "total_documents": len(docs),
        "unique_works": len(work_groups),
        "extra_volumes": skipped_count,
        "selection_rule": "first document per main_work_id (filename prefix before first underscore)",
        "work_volume_counts": {
            wid: len(wdocs) for wid, wdocs in sorted(work_groups.items())
        },
    }
    with open(args.output_dir / "dedupe_report.json", "w") as f:
        json.dump(dedupe_report, f, ensure_ascii=False, indent=2)

    with open(args.output_dir / "data_config.json", "w") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "max_length": args.max_length,
                "stride": args.stride,
                "boundary_radius": args.boundary_radius,
                "neg_sample_ratio": args.neg_sample_ratio,
                "num_unique_works": len(work_groups),
                "num_train_works": len(train_wids),
                "num_val_works": len(val_wids),
                "num_test_works": len(test_wids),
                "num_train_docs": len(train_docs),
                "num_val_docs": len(val_docs),
                "num_test_docs": len(test_docs),
                "raw_train_windows": len(train_docs),
                "num_train_windows": len(train_examples),
                "num_val_windows": len(val_examples),
                "num_test_windows": len(test_examples),
                "label_b_count": train_b,
                "label_o_count": train_o,
                "raw_label_b_count": raw_train_b,
                "raw_label_o_count": raw_train_o,
            },
            f,
            indent=2,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
