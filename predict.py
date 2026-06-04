"""
Run inference on raw Tibetan text files.

Loads a trained mmBERT model and predicts text boundaries in new documents.
Outputs boundary positions and optionally annotated text with <b> markers.

Usage:
    python predict.py input.txt
    python predict.py input.txt --output boundaries.json
    python predict.py input.txt --annotated output_annotated.txt
    python predict.py docs_dir/ --output results/
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from config import (
    CHECKPOINTS_DIR,
    LABEL_B,
    MAX_SEQ_LENGTH,
    STRIDE,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_boundaries(
    text: str,
    model,
    tokenizer,
    device: torch.device,
    max_length: int = MAX_SEQ_LENGTH,
    stride: int = STRIDE,
    threshold: float = 0.5,
) -> list[dict]:
    """
    Predict boundary positions in a text string.

    Returns list of dicts with 'position', 'confidence', and 'context'.
    """
    model.eval()

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        max_length=max_length,
        stride=stride,
        truncation=True,
        padding=False,
    )

    # Track the maximum B probability for each character position across windows
    char_scores = {}

    with torch.no_grad():
        for win_idx in range(len(encoding["input_ids"])):
            input_ids = torch.tensor(
                [encoding["input_ids"][win_idx]], dtype=torch.long
            ).to(device)
            attention_mask = torch.tensor(
                [encoding["attention_mask"][win_idx]], dtype=torch.long
            ).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[0, :, LABEL_B]
            probs = probs.cpu().numpy()

            offsets = encoding["offset_mapping"][win_idx]
            for token_idx, (start, end) in enumerate(offsets):
                if start == 0 and end == 0:
                    continue
                prob = float(probs[token_idx])
                if start not in char_scores or prob > char_scores[start]:
                    char_scores[start] = prob

    # Collect positions above threshold
    raw_positions = [
        (pos, score) for pos, score in char_scores.items() if score >= threshold
    ]
    raw_positions.sort(key=lambda x: x[0])

    # Merge nearby predictions (within 20 chars, keep highest confidence)
    merged = []
    merge_window = 20
    for pos, score in raw_positions:
        if merged and pos - merged[-1][0] < merge_window:
            if score > merged[-1][1]:
                merged[-1] = (pos, score)
        else:
            merged.append((pos, score))

    # Build results with context
    boundaries = []
    for pos, score in merged:
        ctx_start = max(0, pos - 50)
        ctx_end = min(len(text), pos + 50)
        context = text[ctx_start:pos] + " |BOUNDARY| " + text[pos:ctx_end]

        boundaries.append(
            {
                "position": pos,
                "confidence": round(score, 4),
                "context": context,
            }
        )

    return boundaries


def annotate_text(text: str, boundaries: list[dict]) -> str:
    """Insert <b> markers at boundary positions."""
    positions = sorted([b["position"] for b in boundaries], reverse=True)
    result = text
    for pos in positions:
        result = result[:pos] + "<b>" + result[pos:]
    return result


def process_file(
    input_path: Path,
    model,
    tokenizer,
    device,
    max_length: int,
    stride: int,
    threshold: float,
) -> dict:
    text = input_path.read_text(encoding="utf-8")
    boundaries = predict_boundaries(
        text, model, tokenizer, device, max_length, stride, threshold
    )
    return {
        "file": str(input_path),
        "text_length": len(text),
        "num_boundaries": len(boundaries),
        "boundaries": boundaries,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict text boundaries")
    parser.add_argument("input", type=Path, help="Input text file or directory")
    parser.add_argument("--model", type=str, default=str(CHECKPOINTS_DIR / "best"))
    parser.add_argument("--output", type=Path, default=None, help="Output JSON file or directory")
    parser.add_argument("--annotated", type=Path, default=None, help="Output annotated text file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for boundaries")
    parser.add_argument("--max-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--stride", type=int, default=STRIDE)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    print(f"Loading model from {args.model}...")
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("  Model loaded.")

    if args.input.is_dir():
        input_files = sorted(args.input.glob("*.txt"))
        print(f"\nProcessing {len(input_files)} files from {args.input}...")

        all_results = []
        for fpath in tqdm(input_files, desc="Processing"):
            result = process_file(
                fpath, model, tokenizer, device,
                args.max_length, args.stride, args.threshold,
            )
            all_results.append(result)
            print(
                f"  {fpath.name}: {result['num_boundaries']} boundaries "
                f"in {result['text_length']} chars"
            )

        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            for result in all_results:
                fname = Path(result["file"]).stem
                out_path = args.output / f"{fname}_boundaries.json"
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {args.output}")
        else:
            for result in all_results:
                print(f"\n{'=' * 60}")
                print(f"File: {result['file']}")
                print(f"Boundaries: {result['num_boundaries']}")
                for b in result["boundaries"]:
                    print(f"  pos={b['position']} conf={b['confidence']:.3f}")
    else:
        print(f"\nProcessing {args.input}...")
        result = process_file(
            args.input, model, tokenizer, device,
            args.max_length, args.stride, args.threshold,
        )

        print(f"\nFound {result['num_boundaries']} boundaries "
              f"in {result['text_length']} chars")
        print()

        for b in result["boundaries"]:
            print(f"  pos={b['position']:>8}  conf={b['confidence']:.3f}  "
                  f"{b['context'][:80]}...")

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {args.output}")

        if args.annotated:
            text = args.input.read_text(encoding="utf-8")
            annotated = annotate_text(text, result["boundaries"])
            args.annotated.parent.mkdir(parents=True, exist_ok=True)
            args.annotated.write_text(annotated, encoding="utf-8")
            print(f"Annotated text saved to {args.annotated}")


if __name__ == "__main__":
    main()
