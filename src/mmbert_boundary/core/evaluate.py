"""Evaluate a trained mmBERT boundary detector.

Converts token-level B predictions back to character positions, then applies
tolerance-based matching against ground truth boundaries. Generates a detailed
report with per-document and aggregate metrics.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from mmbert_boundary.config import (
    ANNOTATED_DATA_DIR,
    BENCHMARK_DIR,
    CHECKPOINTS_DIR,
    EVAL_BATCH_SIZE,
    LABEL_B,
    MODEL_NAME,
    NUM_LABELS,
    OUTPUT_DIR,
    PROCESSED_DIR,
    TOLERANCE_CHARS,
)
from mmbert_boundary.core.predict import annotate_text, predict_boundaries
from mmbert_boundary.utils.collate import collate_eval
from mmbert_boundary.utils.device import get_device


def predict_with_offsets(
    model: AutoModelForTokenClassification,
    dataset,
    device: torch.device,
    tokenizer: AutoTokenizer,
    batch_size: int,
) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    """Run prediction and map B tokens back to character positions.

    Re-tokenizes each source document with offset mappings to convert
    token-level B predictions to character offsets, then compares against
    the ground-truth breakpoints from the source JSON files.

    Args:
        model: Loaded token-classification model.
        dataset: Tokenised test/benchmark HuggingFace Dataset.
        device: Inference device.
        tokenizer: Matching fast tokenizer.
        batch_size: Batch size for the DataLoader.

    Returns:
        Tuple of ``(doc_pred_chars, doc_true_chars)`` — dicts mapping
        ``doc_id`` to sets of predicted/true character offsets.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_eval, num_workers=0)

    all_results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            doc_ids = batch["doc_ids"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, :, LABEL_B]
            preds = outputs.logits.argmax(dim=-1)

            for i in range(len(doc_ids)):
                all_results.append({
                    "doc_id": doc_ids[i],
                    "input_ids": input_ids[i].cpu().tolist(),
                    "preds": preds[i].cpu().tolist(),
                    "b_probs": probs[i].cpu().tolist(),
                    "labels": (
                        batch["labels"][i].tolist()
                        if isinstance(batch["labels"][i], torch.Tensor)
                        else list(batch["labels"][i])
                    ),
                    "attention_mask": attention_mask[i].cpu().tolist(),
                })

    doc_pred_chars: dict[str, set[int]] = defaultdict(set)
    doc_true_chars: dict[str, set[int]] = defaultdict(set)

    doc_results: dict[str, list] = defaultdict(list)
    for r in all_results:
        doc_results[r["doc_id"]].append(r)

    for doc_id, windows in tqdm(doc_results.items(), desc="Mapping to chars"):
        json_path = ANNOTATED_DATA_DIR / f"{doc_id}.json"
        if not json_path.exists():
            continue
        payload = json.load(open(json_path, encoding="utf-8"))
        text = unicodedata.normalize("NFC", payload.get("content", ""))
        segments = payload.get("segments", [])
        true_breakpoints = {
            seg["span_start"]
            for seg in segments[1:]
            if seg.get("span_start") is not None
        }
        doc_true_chars[doc_id] = true_breakpoints

        win_max_len = len(windows[0]["input_ids"])
        win_stride = min(128, win_max_len - 3)
        data_config_path = PROCESSED_DIR / "data_config.json"
        if data_config_path.exists():
            with open(data_config_path) as cf:
                dc = json.load(cf)
                win_stride = min(dc.get("stride", win_stride), win_max_len - 3)

        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            max_length=win_max_len,
            stride=win_stride,
            truncation=True,
            padding=False,
        )

        for win_idx, window in enumerate(windows):
            if win_idx >= len(encoding["offset_mapping"]):
                break
            offsets = encoding["offset_mapping"][win_idx]

            for token_idx, pred in enumerate(window["preds"]):
                if token_idx >= len(offsets):
                    break
                if window["labels"][token_idx] == -100:
                    continue
                start, end = offsets[token_idx]
                if start == 0 and end == 0:
                    continue
                if pred == LABEL_B:
                    doc_pred_chars[doc_id].add(start)

    return doc_pred_chars, doc_true_chars


def tolerance_match(predicted: set[int], true: set[int], tolerance: int) -> dict:
    """Match predicted positions to true positions with character tolerance.

    Args:
        predicted: Set of predicted boundary character offsets.
        true: Set of ground-truth boundary character offsets.
        tolerance: Maximum allowed distance for a match.

    Returns:
        Dict with precision, recall, f1, tp, fp, fn, total_predicted,
        total_true, matches, false_positives, false_negatives.
    """
    pred_sorted = sorted(predicted)
    true_sorted = sorted(true)

    matched_pred: set[int] = set()
    matched_true: set[int] = set()
    matches = []

    for t in true_sorted:
        best_dist = tolerance + 1
        best_p = None
        for p in pred_sorted:
            if p in matched_pred:
                continue
            dist = abs(p - t)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_p = p
        if best_p is not None:
            matched_pred.add(best_p)
            matched_true.add(t)
            matches.append({"true": t, "pred": best_p, "distance": best_dist})

    tp = len(matches)
    fp = len(predicted) - tp
    fn = len(true) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_predicted": len(predicted),
        "total_true": len(true),
        "matches": matches,
        "false_positives": sorted(predicted - matched_pred),
        "false_negatives": sorted(true - matched_true),
    }


def evaluate_benchmark_with_inference(
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    tolerance: int,
    threshold: float,
    save_inference_dir: Path | None,
) -> tuple[dict[str, dict], dict[str, str]]:
    """Evaluate on benchmark docs using weighted-window inference.

    For each document in ``benchmark_meta.json``, reads the source text from
    the annotated data directory, runs ``predict_boundaries``, and compares
    against ground truth.

    Args:
        model: Loaded token classification model.
        tokenizer: Matching tokenizer.
        device: Torch device.
        tolerance: Char-level tolerance for matching.
        threshold: Confidence threshold for boundary predictions.
        save_inference_dir: If set, save annotated files with ``<b>`` markers here.

    Returns:
        Tuple of ``(per_doc_results, doc_texts)`` where ``doc_texts`` maps
        ``doc_id`` to the original text.
    """
    meta_path = BENCHMARK_DIR / "benchmark_meta.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    if save_inference_dir is not None:
        save_inference_dir.mkdir(parents=True, exist_ok=True)

    per_doc_results: dict[str, dict] = {}
    doc_texts: dict[str, str] = {}

    for doc_info in tqdm(meta["docs"], desc="Evaluating benchmark docs"):
        doc_id = doc_info["doc_id"]
        json_path = ANNOTATED_DATA_DIR / f"{doc_id}.json"
        if not json_path.exists():
            print(f"  WARNING: source JSON not found for {doc_id}, skipping")
            continue

        payload = json.load(open(json_path, encoding="utf-8"))
        text = unicodedata.normalize("NFC", payload.get("content", ""))
        segments = payload.get("segments", [])
        doc_texts[doc_id] = text

        true_breakpoints = {
            seg["span_start"]
            for seg in segments[1:]
            if seg.get("span_start") is not None
        }

        boundaries = predict_boundaries(text, model, tokenizer, device, threshold=threshold)
        pred_positions = {b["position"] for b in boundaries}

        result = tolerance_match(pred_positions, true_breakpoints, tolerance)
        per_doc_results[doc_id] = result

        if save_inference_dir is not None:
            annotated = annotate_text(text, boundaries)
            out_path = save_inference_dir / f"{doc_id}.txt"
            out_path.write_text(annotated, encoding="utf-8")

    return per_doc_results, doc_texts


def print_and_save_report(
    per_doc_results: dict[str, dict],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    """Aggregate metrics, print summary, and write ``eval_report.json``.

    Args:
        per_doc_results: Per-document tolerance-match results.
        args: Parsed CLI namespace (used for config section of the report).
        output_path: Destination JSON file.
    """
    total_tp = sum(r["tp"] for r in per_doc_results.values())
    total_fp = sum(r["fp"] for r in per_doc_results.values())
    total_fn = sum(r["fn"] for r in per_doc_results.values())
    total_pred = sum(r["total_predicted"] for r in per_doc_results.values())
    total_true = sum(r["total_true"] for r in per_doc_results.values())

    agg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    agg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    agg_f1 = (
        2 * agg_precision * agg_recall / (agg_precision + agg_recall)
        if (agg_precision + agg_recall) > 0 else 0.0
    )

    doc_f1s = [r["f1"] for r in per_doc_results.values() if r["total_true"] > 0]
    macro_f1 = float(np.mean(doc_f1s)) if doc_f1s else 0.0

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Tolerance:          {args.tolerance} chars")
    print(f"  Documents:          {len(per_doc_results)}")
    print(f"  Total boundaries:   {total_true}")
    print(f"  Total predicted:    {total_pred}")
    print(f"  ─────────────────────────────")
    print(f"  Micro Precision:    {agg_precision:.4f}  ({total_tp}/{total_tp + total_fp})")
    print(f"  Micro Recall:       {agg_recall:.4f}  ({total_tp}/{total_tp + total_fn})")
    print(f"  Micro F1:           {agg_f1:.4f}")
    print(f"  Macro F1:           {macro_f1:.4f}")
    print(f"  ─────────────────────────────")
    print(f"  True Positives:     {total_tp}")
    print(f"  False Positives:    {total_fp}")
    print(f"  False Negatives:    {total_fn}")
    print("=" * 60)

    print("\nPer-document results (sorted by F1, worst first):")
    sorted_docs = sorted(per_doc_results.items(), key=lambda x: x[1]["f1"])
    for doc_id, result in sorted_docs[:10]:
        print(
            f"  {doc_id[:12]}... "
            f"P={result['precision']:.3f} R={result['recall']:.3f} "
            f"F1={result['f1']:.3f} "
            f"({result['tp']}/{result['total_true']} boundaries)"
        )
    if len(sorted_docs) > 10:
        print(f"  ... ({len(sorted_docs) - 10} more documents)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "config": {
            "model": args.model,
            "tolerance": args.tolerance,
            "threshold": args.threshold,
            "on_benchmark": args.on_benchmark,
        },
        "aggregate": {
            "micro_precision": agg_precision,
            "micro_recall": agg_recall,
            "micro_f1": agg_f1,
            "macro_f1": macro_f1,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "total_predicted": total_pred,
            "total_true": total_true,
            "num_documents": len(per_doc_results),
        },
        "per_document": {
            doc_id: {
                k: v
                for k, v in result.items()
                if k not in ("matches", "false_positives", "false_negatives")
            }
            for doc_id, result in per_doc_results.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {output_path}")


def main(argv: list[str] | None = None) -> None:
    """Evaluate boundary detector against ground truth.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(description="Evaluate boundary detector")
    parser.add_argument(
        "--model",
        type=str,
        default=str(CHECKPOINTS_DIR / "best"),
        help="Path to trained model directory (default: %(default)s)",
    )
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--tolerance", type=int, default=TOLERANCE_CHARS,
                        help="Char tolerance for matching (default: %(default)s)")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Confidence threshold for boundary predictions (default: %(default)s)")
    parser.add_argument("--on-benchmark", action="store_true",
                        help="Use benchmark set instead of test set")
    parser.add_argument("--save-inference", type=Path, default=None,
                        help="Directory to save annotated inference output with <b> markers")
    parser.add_argument("--output", type=Path, default=None,
                        help="Path for eval_report.json (default: output/eval_report.json)")
    args = parser.parse_args(argv)

    device = get_device()
    print(f"Device: {device}")

    print(f"Loading model from {args.model}...")
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    output_path = args.output or OUTPUT_DIR / "eval_report.json"

    if args.on_benchmark:
        print(f"\nEvaluating on benchmark (threshold={args.threshold}, tolerance={args.tolerance})...")
        per_doc_results, _ = evaluate_benchmark_with_inference(
            model, tokenizer, device,
            tolerance=args.tolerance,
            threshold=args.threshold,
            save_inference_dir=args.save_inference,
        )
        if args.save_inference is not None:
            print(f"\nAnnotated inference saved to {args.save_inference}/")
        print_and_save_report(per_doc_results, args, output_path)
    else:
        dataset_path = PROCESSED_DIR / "dataset"
        print(f"Loading test dataset from {dataset_path}...")

        ds = load_from_disk(str(dataset_path))
        test_ds = ds["test"]
        print(f"  Windows: {len(test_ds)}")

        print(f"\nRunning predictions (tolerance={args.tolerance} chars)...")
        doc_preds, doc_true = predict_with_offsets(model, test_ds, device, tokenizer, args.batch_size)

        print("\nComputing metrics...")
        per_doc_results = {}
        for doc_id in sorted(set(list(doc_preds.keys()) + list(doc_true.keys()))):
            pred = doc_preds.get(doc_id, set())
            true = doc_true.get(doc_id, set())
            result = tolerance_match(pred, true, args.tolerance)
            per_doc_results[doc_id] = result

        print_and_save_report(per_doc_results, args, output_path)
