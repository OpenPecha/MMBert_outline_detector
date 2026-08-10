"""Fine-tune mmBERT for Tibetan text boundary detection (token classification).

Key features:
  - bf16 mixed precision via torch.cuda.amp
  - Gradient checkpointing to maximise batch size
  - torch.compile for fused kernels
  - Multi-GPU via DataParallel (auto-detected)
  - Focal loss or Asymmetric loss for extreme class imbalance
  - Cosine-with-warmup schedule
  - Time-aware logging (cost tracking)
  - Negative window sampling (drops most all-O windows)
  - Early stopping on val F2 with configurable patience
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from mmbert_boundary.config import (
    CHECKPOINTS_DIR,
    EVAL_BATCH_SIZE,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    GRAD_ACCUMULATION_STEPS,
    LABEL_B,
    LABEL_NAMES,
    LABEL_O,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    MODEL_NAME,
    NEG_SAMPLE_RATIO,
    NUM_EPOCHS,
    NUM_LABELS,
    OUTPUT_DIR,
    POS_WEIGHT,
    PROCESSED_DIR,
    SEED,
    TRAIN_BATCH_SIZE,
    USE_FOCAL_LOSS,
    WANDB_PROJECT,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from mmbert_boundary.utils.collate import collate_train
from mmbert_boundary.utils.device import get_device
from mmbert_boundary.utils.logging import setup_logging
from mmbert_boundary.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for token classification with extreme class imbalance."""

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0, ignore_index: int = -100) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model output of shape (B, T, C).
            targets: Integer label tensor of shape (B, T).

        Returns:
            Scalar loss tensor.
        """
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)

        mask = targets != self.ignore_index
        logits = logits[mask]
        targets = targets[mask]

        if targets.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        log_probs = nn.functional.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        alpha_t = self.alpha[targets]
        focal_weight = alpha_t * (1.0 - target_probs) ** self.gamma

        loss = -focal_weight * target_log_probs
        return loss.mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric loss (Ridnik et al., 2021) for extreme class imbalance.

    Applies separate gamma values for positive (B) and negative (O) tokens:
      - gamma_pos=0  → no down-weighting for rare B tokens
      - gamma_neg>0  → aggressively down-weight confident O predictions
    """

    def __init__(
        self,
        alpha: torch.Tensor,
        gamma_pos: float = 0.0,
        gamma_neg: float = 3.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute asymmetric loss.

        Args:
            logits: Raw model output of shape (B, T, C).
            targets: Integer label tensor of shape (B, T).

        Returns:
            Scalar loss tensor.
        """
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)

        mask = targets != self.ignore_index
        logits = logits[mask]
        targets = targets[mask]

        if targets.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        log_probs = nn.functional.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        alpha_t = self.alpha[targets]
        gamma_t = torch.where(
            targets == LABEL_B,
            torch.full_like(target_probs, self.gamma_pos),
            torch.full_like(target_probs, self.gamma_neg),
        )
        focal_weight = alpha_t * (1.0 - target_probs) ** gamma_t

        loss = -focal_weight * target_log_probs
        return loss.mean()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_label_counts(dataset) -> tuple[int, int]:
    """Count B and O labels in a dataset.

    Args:
        dataset: HuggingFace Dataset with a ``labels`` column.

    Returns:
        Tuple of ``(b_count, o_count)``.
    """
    b_count = o_count = 0
    for labels in dataset["labels"]:
        for lbl in labels:
            if lbl == LABEL_B:
                b_count += 1
            elif lbl == LABEL_O:
                o_count += 1
    return b_count, o_count


def filter_negative_windows(dataset, neg_sample_ratio: float, seed: int = 42):
    """Keep all windows with B labels, subsample windows with only O labels.

    Prefers the precomputed ``has_boundary`` column from prepare-data
    (index filter). Falls back to scanning ``labels`` for older datasets.

    Args:
        dataset: HuggingFace Dataset with ``has_boundary`` and/or ``labels``.
        neg_sample_ratio: Fraction of all-O windows to keep.
        seed: RNG seed.

    Returns:
        Filtered Dataset.
    """
    rng = np.random.default_rng(seed)

    if "has_boundary" in dataset.column_names:
        flags = dataset["has_boundary"]
        positive_indices = [i for i, flag in enumerate(flags) if flag]
        negative_indices = [i for i, flag in enumerate(flags) if not flag]
    else:
        positive_indices = []
        negative_indices = []
        for i, labels in enumerate(dataset["labels"]):
            if any(lbl == LABEL_B for lbl in labels):
                positive_indices.append(i)
            else:
                negative_indices.append(i)

    n_neg_keep = int(len(negative_indices) * neg_sample_ratio)
    if n_neg_keep > 0 and negative_indices:
        kept_negatives = rng.choice(
            negative_indices, size=min(n_neg_keep, len(negative_indices)), replace=False
        ).tolist()
    else:
        kept_negatives = []

    all_indices = sorted(positive_indices + kept_negatives)
    filtered = dataset.select(all_indices)
    print(f"  Negative sampling: {len(dataset):,} → {len(filtered):,} windows "
          f"(kept {len(positive_indices):,} pos + {len(kept_negatives):,}/{len(negative_indices):,} neg)")
    return filtered

def fmt_time(seconds: float) -> str:
    """Format seconds as Hh MM m SS s.

    Args:
        seconds: Duration in seconds.

    Returns:
        Human-readable string like ``1h02m45s``.
    """
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataloader, device, loss_fn, amp_dtype) -> dict:
    """Run the validation loop and return metrics.

    Args:
        model: Model in eval mode.
        dataloader: Validation DataLoader.
        device: Inference device.
        loss_fn: Loss function.
        amp_dtype: AMP dtype (used only on CUDA).

    Returns:
        Dict with ``precision``, ``recall``, ``f1``, ``f2``, ``loss``,
        ``tp``, ``fp``, ``fn``.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    tp = fp = fn = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        loss = loss_fn(logits.float(), labels)
        total_loss += loss.item()
        num_batches += 1

        preds = logits.argmax(dim=-1)
        mask = labels != -100
        p, lbl = preds[mask], labels[mask]
        tp += int(((p == LABEL_B) & (lbl == LABEL_B)).sum())
        fp += int(((p == LABEL_B) & (lbl == LABEL_O)).sum())
        fn += int(((p == LABEL_O) & (lbl == LABEL_B)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    beta = 2.0
    f2 = (
        (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "precision": precision, "recall": recall, "f1": f1, "f2": f2,
        "tp": tp, "fp": fp, "fn": fn,
        "loss": total_loss / max(num_batches, 1),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Fine-tune mmBERT on the prepared dataset.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(description="Train mmBERT boundary detector")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR / "dataset")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--grad-accumulation", type=int, default=GRAD_ACCUMULATION_STEPS)
    parser.add_argument("--pos-weight", type=float, default=POS_WEIGHT,
                        help="Manual B-class weight; omit to compute from data")
    parser.add_argument("--focal-loss", action="store_true", default=USE_FOCAL_LOSS)
    parser.add_argument("--no-focal-loss", dest="focal_loss", action="store_false")
    parser.add_argument("--focal-gamma", type=float, default=FOCAL_GAMMA)
    parser.add_argument("--focal-alpha", type=float, default=FOCAL_ALPHA)
    parser.add_argument("--asymmetric-loss", action="store_true", default=False,
                        help="Use asymmetric loss instead of focal loss")
    parser.add_argument("--asl-gamma-neg", type=float, default=3.0,
                        help="Negative gamma for asymmetric loss")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=600)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping: halt after N evals without F2 improvement")
    parser.add_argument("--neg-sample-ratio", type=float, default=NEG_SAMPLE_RATIO,
                        help="Fraction of all-O windows to keep")
    parser.add_argument("--warmup-ratio", type=float, default=None,
                        help="Override warmup ratio")
    parser.add_argument("--compile", action="store_true", default=True,
                        help="Use torch.compile (needs PyTorch 2.0+)")
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--cost-per-hour", type=float, default=0.0,
                        help="$/hr for cost tracking in logs")
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable Weights & Biases experiment tracking")
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT,
                        help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (auto-generated if omitted)")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity (team or username; uses default if omitted)")
    args = parser.parse_args(argv)

    setup_logging(args.output_dir)
    set_seed(SEED)
    device = get_device()

    print(f"Device: {device}")
    if device.type == "cuda":
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} — {props.total_memory / 1024**3:.1f} GB")
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"  AMP dtype: {amp_dtype}")
        torch.backends.cudnn.benchmark = True
    else:
        amp_dtype = torch.float16

    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 1

    print(f"\nLoading dataset from {args.data_dir}...")
    ds = load_from_disk(str(args.data_dir))
    train_ds, val_ds = ds["train"], ds["validation"]
    print(f"  Train: {len(train_ds):,} windows")
    print(f"  Val:   {len(val_ds):,} windows")

    if args.neg_sample_ratio < 1.0:
        print(f"\nApplying negative window sampling (ratio={args.neg_sample_ratio})...")
        train_ds = filter_negative_windows(train_ds, args.neg_sample_ratio, seed=SEED)

    print(f"\nLoading model: {args.model_name}")
    if args.resume:
        print(f"  Resuming from {args.resume}")
        model = AutoModelForTokenClassification.from_pretrained(args.resume, num_labels=NUM_LABELS)
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name, num_labels=NUM_LABELS,
            id2label={str(i): lbl for i, lbl in enumerate(LABEL_NAMES)},
            label2id={lbl: i for i, lbl in enumerate(LABEL_NAMES)},
        )

    model.gradient_checkpointing_enable()

    if device.type == "cuda":
        model = model.to(device)
        if args.compile:
            try:
                model = torch.compile(model)
                print("  torch.compile: enabled")
            except Exception as e:
                print(f"  torch.compile: skipped ({e})")

        if n_gpu > 1:
            model = nn.DataParallel(model)
            print(f"  DataParallel: {n_gpu} GPUs")
    else:
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("\nComputing label distribution...")
    b_count, o_count = compute_label_counts(train_ds)
    ratio = o_count / max(b_count, 1)
    print(f"  B={b_count:,}  O={o_count:,}  ratio=1:{ratio:.0f}")

    alpha_b = args.focal_alpha if args.focal_alpha is not None else min(ratio / (1 + ratio), 0.99)
    alpha_o = 1.0 - alpha_b
    alpha_tensor = torch.tensor([alpha_o, alpha_b], dtype=torch.float32, device=device)

    if args.asymmetric_loss:
        loss_fn = AsymmetricLoss(alpha=alpha_tensor, gamma_pos=0.0, gamma_neg=args.asl_gamma_neg)
        print(f"  Asymmetric Loss: gamma_pos=0.0, gamma_neg={args.asl_gamma_neg}, "
              f"alpha=[O={alpha_o:.4f}, B={alpha_b:.4f}]")
    elif args.focal_loss:
        loss_fn = FocalLoss(alpha=alpha_tensor, gamma=args.focal_gamma)
        print(f"  Focal Loss: gamma={args.focal_gamma}, alpha=[O={alpha_o:.4f}, B={alpha_b:.4f}]")
    else:
        pw = args.pos_weight if args.pos_weight is not None else min(math.sqrt(ratio), 500.0)
        class_weights = torch.tensor([1.0, pw], dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        print(f"  Weighted CE: pos_weight={pw:.1f}")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_train, num_workers=args.workers,
        pin_memory=pin, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=collate_train, num_workers=args.workers,
        pin_memory=pin, persistent_workers=args.workers > 0,
    )

    raw_model = model
    if isinstance(raw_model, torch._dynamo.eval_frame.OptimizedModule):
        raw_model = raw_model._orig_mod
    if isinstance(raw_model, nn.DataParallel):
        raw_model = raw_model.module

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16)
    )

    steps_per_epoch = len(train_loader) // args.grad_accumulation
    total_steps = steps_per_epoch * args.epochs
    if args.warmup_ratio is not None:
        warmup_ratio = args.warmup_ratio
    elif args.resume:
        warmup_ratio = 0.02
    else:
        warmup_ratio = WARMUP_RATIO
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    eff_batch = args.batch_size * args.grad_accumulation * n_gpu
    print(f"\nTraining config:")
    print(f"  Epochs:               {args.epochs}")
    print(f"  Batch size per GPU:   {args.batch_size}")
    print(f"  Grad accumulation:    {args.grad_accumulation}")
    print(f"  GPUs:                 {n_gpu}")
    print(f"  Effective batch size: {eff_batch}")
    print(f"  Learning rate:        {args.lr}")
    print(f"  Steps/epoch:          {steps_per_epoch:,}")
    print(f"  Total optim steps:    {total_steps:,}")
    print(f"  Warmup steps:         {warmup_steps:,}")
    print(f"  Neg sample ratio:     {args.neg_sample_ratio}")
    print(f"  Early stop patience:  {args.patience} evals")
    if args.cost_per_hour:
        print(f"  Cost tracking:        ${args.cost_per_hour:.2f}/hr")

    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    _wandb_run = None
    if args.wandb:
        try:
            import wandb as _wandb_lib
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Run `pip install wandb` or reinstall: `pip install -e .`"
            ) from exc
        _wandb_run = _wandb_lib.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config={
                "model_name": args.model_name,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "grad_accumulation": args.grad_accumulation,
                "effective_batch_size": eff_batch,
                "learning_rate": args.lr,
                "warmup_ratio": warmup_ratio,
                "weight_decay": WEIGHT_DECAY,
                "max_grad_norm": MAX_GRAD_NORM,
                "focal_loss": args.focal_loss,
                "focal_gamma": args.focal_gamma,
                "focal_alpha": args.focal_alpha,
                "asymmetric_loss": args.asymmetric_loss,
                "asl_gamma_neg": args.asl_gamma_neg,
                "neg_sample_ratio": args.neg_sample_ratio,
                "eval_steps": args.eval_steps,
                "save_steps": args.save_steps,
                "patience": args.patience,
                "data_dir": str(args.data_dir),
                "output_dir": str(args.output_dir),
                "seed": SEED,
                "n_gpu": n_gpu,
                "device": str(device),
                "resume": args.resume,
                "b_count": b_count,
                "o_count": o_count,
                "bo_ratio": f"1:{ratio:.0f}",
                "train_windows": len(train_ds),
                "val_windows": len(val_ds),
            },
        )
        print(f"  W&B run: {_wandb_run.url}")

    best_f1 = 0.0
    global_step = 0
    evals_without_improvement = 0
    training_log = []
    train_start = time.time()

    print("\n" + "=" * 70)
    print("Starting training")
    if args.patience:
        print(f"  Early stopping: patience={args.patience} evals")
    print("=" * 70)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"
            ):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits.float(), labels)
                loss = loss / args.grad_accumulation

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accumulation

            if (step + 1) % args.grad_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                elapsed = time.time() - train_start
                it_per_sec = (step + 1) / (time.time() - epoch_start)
                postfix = {
                    "loss": f"{loss.item() * args.grad_accumulation:.6f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.1e}",
                    "it/s": f"{it_per_sec:.1f}",
                }
                if args.cost_per_hour:
                    postfix["$"] = f"{elapsed / 3600 * args.cost_per_hour:.2f}"
                pbar.set_postfix(postfix)

                if _wandb_run is not None:
                    _wandb_run.log({
                        "train/loss": loss.item() * args.grad_accumulation,
                        "train/lr": scheduler.get_last_lr()[0],
                    }, step=global_step)

                if global_step % args.eval_steps == 0:
                    metrics = evaluate(model, val_loader, device, loss_fn, amp_dtype)
                    log_entry = {
                        "step": global_step, "epoch": epoch + 1,
                        "elapsed_s": elapsed,
                        "train_loss": epoch_loss / (step + 1),
                        **{f"val_{k}": v for k, v in metrics.items()},
                    }
                    if args.cost_per_hour:
                        log_entry["cost_usd"] = elapsed / 3600 * args.cost_per_hour
                    training_log.append(log_entry)

                    if _wandb_run is not None:
                        _wm: dict = {
                            "val/loss": metrics["loss"],
                            "val/precision": metrics["precision"],
                            "val/recall": metrics["recall"],
                            "val/f1": metrics["f1"],
                            "val/f2": metrics["f2"],
                            "val/tp": metrics["tp"],
                            "val/fp": metrics["fp"],
                            "val/fn": metrics["fn"],
                            "train/loss_avg": epoch_loss / (step + 1),
                            "epoch": epoch + 1,
                        }
                        if args.cost_per_hour:
                            _wm["cost_usd"] = log_entry.get("cost_usd", 0)
                        _wandb_run.log(_wm, step=global_step)

                    cost_str = f"  ${log_entry.get('cost_usd', 0):.2f}" if args.cost_per_hour else ""
                    print(
                        f"\n  Step {global_step} [{fmt_time(elapsed)}{cost_str}]: "
                        f"val_loss={metrics['loss']:.4f}, "
                        f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
                        f"F1={metrics['f1']:.3f}, F2={metrics['f2']:.3f} "
                        f"(tp={metrics['tp']}, fp={metrics['fp']}, fn={metrics['fn']})"
                    )

                    if metrics["f2"] > best_f1:
                        best_f1 = metrics["f2"]
                        evals_without_improvement = 0
                        best_dir = checkpoints_dir / "best"
                        raw_model.save_pretrained(str(best_dir))
                        tokenizer.save_pretrained(str(best_dir))
                        print(f"  >>> New best F2={best_f1:.4f} — saved to {best_dir}")
                    else:
                        evals_without_improvement += 1
                        print(f"  No improvement ({evals_without_improvement}/{args.patience})")

                    if args.patience and evals_without_improvement >= args.patience:
                        print(f"\n  *** Early stopping triggered (no F2 improvement for "
                              f"{args.patience} evals). Best F2={best_f1:.4f} ***")
                        break

                    model.train()

                if global_step % args.save_steps == 0:
                    ckpt_dir = checkpoints_dir / f"checkpoint-{global_step}"
                    raw_model.save_pretrained(str(ckpt_dir))
                    tokenizer.save_pretrained(str(ckpt_dir))

        if args.patience and evals_without_improvement >= args.patience:
            break

        epoch_elapsed = time.time() - epoch_start
        avg_loss = epoch_loss / len(train_loader)
        total_elapsed = time.time() - train_start
        print(f"\nEpoch {epoch+1} done in {fmt_time(epoch_elapsed)}  |  avg loss: {avg_loss:.6f}")

        print("Running end-of-epoch validation...")
        metrics = evaluate(model, val_loader, device, loss_fn, amp_dtype)
        log_entry = {
            "step": global_step, "epoch": epoch + 1, "epoch_end": True,
            "elapsed_s": total_elapsed, "train_loss": avg_loss,
            **{f"val_{k}": v for k, v in metrics.items()},
        }
        if args.cost_per_hour:
            log_entry["cost_usd"] = total_elapsed / 3600 * args.cost_per_hour
        training_log.append(log_entry)

        if _wandb_run is not None:
            _wm = {
                "val/loss": metrics["loss"],
                "val/precision": metrics["precision"],
                "val/recall": metrics["recall"],
                "val/f1": metrics["f1"],
                "val/f2": metrics["f2"],
                "val/tp": metrics["tp"],
                "val/fp": metrics["fp"],
                "val/fn": metrics["fn"],
                "train/loss_epoch": avg_loss,
                "epoch": epoch + 1,
            }
            if args.cost_per_hour:
                _wm["cost_usd"] = log_entry.get("cost_usd", 0)
            _wandb_run.log(_wm, step=global_step)

        cost_str = f"  ${log_entry.get('cost_usd', 0):.2f}" if args.cost_per_hour else ""
        print(
            f"  Val: loss={metrics['loss']:.4f}, "
            f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
            f"F1={metrics['f1']:.3f}, F2={metrics['f2']:.3f}{cost_str}"
        )

        if metrics["f2"] > best_f1:
            best_f1 = metrics["f2"]
            best_dir = checkpoints_dir / "best"
            raw_model.save_pretrained(str(best_dir))
            tokenizer.save_pretrained(str(best_dir))
            print(f"  >>> New best F2={best_f1:.4f}")

    final_dir = checkpoints_dir / "final"
    raw_model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    total_time = time.time() - train_start
    with open(args.output_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    if _wandb_run is not None:
        _wandb_run.summary["best_f2"] = best_f1
        _wandb_run.summary["total_time_s"] = total_time
        _wandb_run.summary["best_checkpoint"] = str(checkpoints_dir / "best")
        _wandb_run.finish()

    print("\n" + "=" * 70)
    print(f"Training complete in {fmt_time(total_time)}")
    if args.cost_per_hour:
        print(f"Estimated cost: ${total_time / 3600 * args.cost_per_hour:.2f}")
    print(f"Best F2: {best_f1:.4f}")
    print(f"Best model:  {checkpoints_dir / 'best'}")
    print(f"Final model: {final_dir}")
    print("=" * 70)
