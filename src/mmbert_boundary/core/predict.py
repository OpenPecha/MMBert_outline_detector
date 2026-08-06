"""Inference on raw Tibetan text files.

Loads a trained mmBERT model and predicts text boundaries in new documents.
Outputs boundary positions and optionally annotated text with <b> markers.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from mmbert_boundary.config import CHECKPOINTS_DIR, LABEL_B, MAX_SEQ_LENGTH, STRIDE
from mmbert_boundary.utils.device import get_device


def predict_boundaries(
    text: str,
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int = MAX_SEQ_LENGTH,
    stride: int = STRIDE,
    threshold: float = 0.5,
) -> list[dict]:
    """Predict boundary positions in a text string.

    Accumulates weighted B probabilities for each character position across
    sliding windows; tokens near the window centre receive full weight (1.0),
    tokens near edges taper to 0.5 to reduce noise from truncated context.

    Args:
        text: Full document text to segment.
        model: Loaded token-classification model.
        tokenizer: Matching fast tokenizer.
        device: Device to run inference on.
        max_length: Sliding-window size in tokens.
        stride: Token overlap between consecutive windows.
        threshold: Minimum weighted probability to emit a boundary.

    Returns:
        List of dicts with ``position`` (char offset), ``confidence``,
        and ``context`` (text snippet around the boundary).
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

    char_score_sum: dict[int, float] = {}
    char_weight_sum: dict[int, float] = {}

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
            n_tokens = len(offsets)
            for token_idx, (start, end) in enumerate(offsets):
                if start == 0 and end == 0:
                    continue
                centre_dist = abs(token_idx - n_tokens / 2.0) / (n_tokens / 2.0)
                weight = 1.0 - 0.5 * centre_dist
                prob = float(probs[token_idx])
                char_score_sum[start] = char_score_sum.get(start, 0.0) + weight * prob
                char_weight_sum[start] = char_weight_sum.get(start, 0.0) + weight

    char_scores = {
        pos: char_score_sum[pos] / char_weight_sum[pos]
        for pos in char_score_sum
    }

    raw_positions = [
        (pos, score) for pos, score in char_scores.items() if score >= threshold
    ]
    raw_positions.sort(key=lambda x: x[0])

    merged: list[tuple[int, float]] = []
    merge_window = 50
    for pos, score in raw_positions:
        if merged and pos - merged[-1][0] < merge_window:
            if score > merged[-1][1]:
                merged[-1] = (pos, score)
        else:
            merged.append((pos, score))

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
    """Insert ``<b>`` markers at boundary positions.

    Args:
        text: Original document text.
        boundaries: Boundary dicts as returned by :func:`predict_boundaries`.

    Returns:
        Text with ``<b>`` inserted at each predicted boundary position.
    """
    positions = sorted([b["position"] for b in boundaries], reverse=True)
    result = text
    for pos in positions:
        result = result[:pos] + "<b>" + result[pos:]
    return result


def postprocess_annotations(text: str) -> str:
    """Nudge ``<b>`` markers to correct Tibetan syllable/word boundaries.

    Rules applied (in order):

    Right-shift rules – move ``<b>`` past trailing punctuation:
      - Past shad (།) clusters and leading spaces.
      - Past a closing parenthesis.

    Left-shift rules – move ``<b>`` before prefix/opening characters:
      - Before ༄ (Yig mgo, U+0F04).
      - Before ༈ (Yig mgo mdun ma, U+0F08) and trailing spaces.
      - Before a Tibetan prefix consonant (འ མ ག ད བ) split from its syllable.

    Args:
        text: Annotated text containing ``<b>`` boundary markers.

    Returns:
        Text with ``<b>`` markers repositioned to correct boundaries.
    """
    text = re.sub(r"<b>( *།[ །]*)", r"\1<b>", text)
    text = re.sub(r"<b>(\))", r"\1<b>", text)
    text = re.sub(r"(༄)<b>", r"<b>\1", text)
    text = re.sub(r"(༈[ ]*)<b>", r"<b>\1", text)
    text = re.sub(r"([འམགདབ])<b>([ཀ-ྼ])", r"<b>\1\2", text)
    return text


def _process_file(
    input_path: Path,
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int,
    stride: int,
    threshold: float,
) -> dict:
    """Run prediction on a single file and return a result dict.

    Args:
        input_path: Path to the input ``.txt`` file.
        model: Loaded token-classification model.
        tokenizer: Matching fast tokenizer.
        device: Inference device.
        max_length: Sliding-window size in tokens.
        stride: Token overlap.
        threshold: Boundary confidence threshold.

    Returns:
        Dict with ``file``, ``text_length``, ``num_boundaries``,
        and ``boundaries``.
    """
    text = input_path.read_text(encoding="utf-8")
    boundaries = predict_boundaries(text, model, tokenizer, device, max_length, stride, threshold)
    return {
        "file": str(input_path),
        "text_length": len(text),
        "num_boundaries": len(boundaries),
        "boundaries": boundaries,
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for inference.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(description="Predict text boundaries in Tibetan documents")
    parser.add_argument("input", type=Path, help="Input .txt file or directory of .txt files")
    parser.add_argument(
        "--model",
        type=str,
        default=str(CHECKPOINTS_DIR / "best"),
        help="Path to a trained model directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file (single input) or directory of JSON files (batch)",
    )
    parser.add_argument(
        "--annotated",
        type=Path,
        default=None,
        help="Output annotated .txt file (single) or directory (batch)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Confidence threshold for boundary predictions (default: %(default)s)",
    )
    parser.add_argument("--max-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--stride", type=int, default=STRIDE)
    args = parser.parse_args(argv)

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

        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
        if args.annotated:
            args.annotated.mkdir(parents=True, exist_ok=True)

        all_results = []
        for fpath in tqdm(input_files, desc="Processing"):
            result = _process_file(
                fpath, model, tokenizer, device,
                args.max_length, args.stride, args.threshold,
            )
            all_results.append(result)
            print(
                f"  {fpath.name}: {result['num_boundaries']} boundaries "
                f"in {result['text_length']} chars"
            )

            if args.annotated:
                text = fpath.read_text(encoding="utf-8")
                annotated = annotate_text(text, result["boundaries"])
                annotated = postprocess_annotations(annotated)
                out_path = args.annotated / fpath.name
                out_path.write_text(annotated, encoding="utf-8")

        if args.output:
            for result in all_results:
                fname = Path(result["file"]).stem
                out_path = args.output / f"{fname}_boundaries.json"
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nBoundary JSON files saved to {args.output}")

        if args.annotated:
            print(f"Annotated text files saved to {args.annotated}")

        if not args.output and not args.annotated:
            for result in all_results:
                print(f"\n{'=' * 60}")
                print(f"File: {result['file']}")
                print(f"Boundaries: {result['num_boundaries']}")
                for b in result["boundaries"]:
                    print(f"  pos={b['position']} conf={b['confidence']:.3f}")
    else:
        print(f"\nProcessing {args.input}...")
        result = _process_file(
            args.input, model, tokenizer, device,
            args.max_length, args.stride, args.threshold,
        )

        print(f"\nFound {result['num_boundaries']} boundaries in {result['text_length']} chars")
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
            annotated = postprocess_annotations(annotated)
            args.annotated.parent.mkdir(parents=True, exist_ok=True)
            args.annotated.write_text(annotated, encoding="utf-8")
            print(f"Annotated text saved to {args.annotated}")
