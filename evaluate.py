"""
Evaluate a trained mmBERT boundary detector.

Converts token-level B predictions back to character positions, then applies
tolerance-based matching against ground truth boundaries. Generates a detailed
report with per-document and aggregate metrics.

Usage:
    python evaluate.py --model output/checkpoints/best
    python evaluate.py --model output/checkpoints/best --tolerance 20 --on-benchmark
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from config import (
    ANNOTATIONS_FILE,
    BENCHMARK_DIR,
    CHECKPOINTS_DIR,
    DOCUMENTS_DIR,
    EVAL_BATCH_SIZE,
    LABEL_B,
    MODEL_NAME,
    NUM_LABELS,
    OUTPUT_DIR,
    PROCESSED_DIR,
    TOLERANCE_CHARS,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_fn(batch):
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    attention_mask = []
    labels = []
    doc_ids = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [0] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
        doc_ids.append(item["doc_id"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "doc_ids": doc_ids,
    }


def predict_boundaries_token_level(model, dataloader, device, tokenizer):
    """
    Run model on all windows and collect per-document boundary char positions.

    For each token predicted as B, we record the character start position from
    the offset mapping. Windows for the same document are merged, deduplicating
    positions within a tolerance.
    """
    model.eval()
    doc_predictions = defaultdict(set)
    doc_true_labels = defaultdict(set)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"]
            doc_ids = batch["doc_ids"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1).cpu().numpy()

            input_ids_cpu = input_ids.cpu()
            for i in range(len(doc_ids)):
                doc_id = doc_ids[i]
                seq_input_ids = input_ids_cpu[i].tolist()
                seq_labels = labels_batch[i]
                if isinstance(seq_labels, torch.Tensor):
                    seq_labels = seq_labels.tolist()
                else:
                    seq_labels = list(seq_labels)
                seq_preds = preds[i]

                # We need offset mapping to convert back to char positions.
                # Re-tokenize the text for this doc to get offsets.
                # Store predicted token indices that are B.
                for token_idx in range(len(seq_preds)):
                    if token_idx >= len(seq_labels):
                        break
                    label = seq_labels[token_idx]
                    if label == -100:
                        continue
                    if seq_preds[token_idx] == LABEL_B:
                        doc_predictions[doc_id].add(token_idx)

    return doc_predictions


def predict_with_offsets(model, dataset, device, tokenizer, batch_size):
    """
    Run prediction and map B tokens back to character positions using
    re-tokenization of the source documents.
    """
    model.eval()

    # Group windows by doc_id and collect predictions
    doc_windows = defaultdict(list)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    all_results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Predicting")):
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
                    "labels": batch["labels"][i].tolist() if isinstance(batch["labels"][i], torch.Tensor) else list(batch["labels"][i]),
                    "attention_mask": attention_mask[i].cpu().tolist(),
                })

    # Now re-tokenize each doc to get offset mappings and align predictions
    doc_pred_chars = defaultdict(set)
    doc_true_chars = defaultdict(set)

    # Group results by doc
    doc_results = defaultdict(list)
    for r in all_results:
        doc_results[r["doc_id"]].append(r)

    annotations = {}
    with open(ANNOTATIONS_FILE) as f:
        annotations = json.load(f)

    for doc_id, windows in tqdm(doc_results.items(), desc="Mapping to chars"):
        doc_path = DOCUMENTS_DIR / f"{doc_id}.txt"
        if not doc_path.exists():
            continue
        text = doc_path.read_text(encoding="utf-8")
        ann = annotations.get(doc_id, {})
        true_breakpoints = set(ann.get("breakpoints", []))
        doc_true_chars[doc_id] = true_breakpoints

        # Re-tokenize the full document with return_offsets_mapping
        win_max_len = len(windows[0]["input_ids"])
        win_stride = min(128, win_max_len - 3)
        # Load data config stride if available
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

        # Match each window's predictions to the re-tokenized offsets
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
    """Match predicted positions to true positions with character tolerance."""
    pred_sorted = sorted(predicted)
    true_sorted = sorted(true)

    matched_pred = set()
    matched_true = set()
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate boundary detector")
    parser.add_argument("--model", type=str, default=str(CHECKPOINTS_DIR / "best"))
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--tolerance", type=int, default=TOLERANCE_CHARS)
    parser.add_argument("--on-benchmark", action="store_true", help="Use benchmark set instead of test set")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    print(f"Loading model from {args.model}...")
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.on_benchmark:
        dataset_path = BENCHMARK_DIR / "dataset"
        print(f"Loading benchmark dataset from {dataset_path}...")
    else:
        dataset_path = PROCESSED_DIR / "dataset"
        print(f"Loading test dataset from {dataset_path}...")

    ds = load_from_disk(str(dataset_path))
    if args.on_benchmark:
        test_ds = ds
    else:
        test_ds = ds["test"]
    print(f"  Windows: {len(test_ds)}")

    print(f"\nRunning predictions (tolerance={args.tolerance} chars)...")
    doc_preds, doc_true = predict_with_offsets(
        model, test_ds, device, tokenizer, args.batch_size
    )

    print("\nComputing metrics...")
    per_doc_results = {}
    all_pred = set()
    all_true = set()

    for doc_id in sorted(set(list(doc_preds.keys()) + list(doc_true.keys()))):
        pred = doc_preds.get(doc_id, set())
        true = doc_true.get(doc_id, set())

        # Offset predictions to global namespace for aggregate calc
        result = tolerance_match(pred, true, args.tolerance)
        per_doc_results[doc_id] = result

    # Aggregate metrics
    total_tp = sum(r["tp"] for r in per_doc_results.values())
    total_fp = sum(r["fp"] for r in per_doc_results.values())
    total_fn = sum(r["fn"] for r in per_doc_results.values())
    total_pred = sum(r["total_predicted"] for r in per_doc_results.values())
    total_true = sum(r["total_true"] for r in per_doc_results.values())

    agg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    agg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    agg_f1 = (
        2 * agg_precision * agg_recall / (agg_precision + agg_recall)
        if (agg_precision + agg_recall) > 0
        else 0.0
    )

    doc_f1s = [r["f1"] for r in per_doc_results.values() if r["total_true"] > 0]
    macro_f1 = np.mean(doc_f1s) if doc_f1s else 0.0

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

    # Per-document breakdown
    print("\nPer-document results (sorted by F1):")
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

    # Save full report
    output_path = args.output or OUTPUT_DIR / "eval_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "config": {
            "model": args.model,
            "tolerance": args.tolerance,
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


if __name__ == "__main__":
    main()
