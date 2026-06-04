"""
Fine-tune mmBERT for Tibetan text boundary detection (token classification).

Optimised for CUDA GPUs (A100/H100). Key features:
  - bf16 mixed precision via torch.cuda.amp
  - Gradient checkpointing to maximise batch size
  - torch.compile for fused kernels
  - Multi-GPU via DataParallel (auto-detected)
  - Focal loss for extreme class imbalance
  - Cosine-with-warmup schedule
  - Time-aware logging (cost tracking)

Usage (single GPU):
    python train.py

Recommended A100 80 GB config:
    python train.py --batch-size 8 --grad-accumulation 2 --epochs 3 \
        --eval-steps 300 --save-steps 600

Resume:
    python train.py --resume output/checkpoints/checkpoint-600
"""

import argparse
import json
import math
import os
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

from config import (
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
    NUM_EPOCHS,
    NUM_LABELS,
    OUTPUT_DIR,
    POS_WEIGHT,
    PROCESSED_DIR,
    SEED,
    TRAIN_BATCH_SIZE,
    USE_FOCAL_LOSS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)


# ── Focal Loss ───────────────────────────────────────────────


class FocalLoss(nn.Module):
    """Focal loss for token classification with extreme class imbalance."""

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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


# ── Helpers ──────────────────────────────────────────────────


def compute_label_counts(dataset) -> tuple[int, int]:
    b_count = o_count = 0
    for labels in dataset["labels"]:
        for l in labels:
            if l == LABEL_B:
                b_count += 1
            elif l == LABEL_O:
                o_count += 1
    return b_count, o_count


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [0] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def fmt_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


# ── Evaluate ─────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model, dataloader, device, loss_fn, amp_dtype) -> dict:
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
        p, l = preds[mask], labels[mask]
        tp += int(((p == LABEL_B) & (l == LABEL_B)).sum())
        fp += int(((p == LABEL_B) & (l == LABEL_O)).sum())
        fn += int(((p == LABEL_O) & (l == LABEL_B)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "loss": total_loss / max(num_batches, 1),
    }


# ── Main ─────────────────────────────────────────────────────


def main():
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
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-steps", type=int, default=300)
    parser.add_argument("--save-steps", type=int, default=600)
    parser.add_argument("--compile", action="store_true", default=True,
                        help="Use torch.compile (needs PyTorch 2.0+)")
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--cost-per-hour", type=float, default=0.0,
                        help="$/hr for cost tracking in logs")
    args = parser.parse_args()

    set_seed(SEED)
    device = get_device()

    # ── Device info ──
    print(f"Device: {device}")
    if device.type == "cuda":
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} — {props.total_mem / 1024**3:.1f} GB")
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"  AMP dtype: {amp_dtype}")
        torch.backends.cudnn.benchmark = True
    else:
        amp_dtype = torch.float16

    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 1

    # ── Data ──
    print(f"\nLoading dataset from {args.data_dir}...")
    ds = load_from_disk(str(args.data_dir))
    train_ds, val_ds = ds["train"], ds["validation"]
    print(f"  Train: {len(train_ds):,} windows")
    print(f"  Val:   {len(val_ds):,} windows")

    # ── Model ──
    print(f"\nLoading model: {args.model_name}")
    if args.resume:
        print(f"  Resuming from {args.resume}")
        model = AutoModelForTokenClassification.from_pretrained(args.resume, num_labels=NUM_LABELS)
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name, num_labels=NUM_LABELS,
            id2label={str(i): l for i, l in enumerate(LABEL_NAMES)},
            label2id={l: i for i, l in enumerate(LABEL_NAMES)},
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

    # ── Loss ──
    print("\nComputing label distribution...")
    b_count, o_count = compute_label_counts(train_ds)
    ratio = o_count / max(b_count, 1)
    print(f"  B={b_count:,}  O={o_count:,}  ratio=1:{ratio:.0f}")

    if args.focal_loss:
        alpha_b = args.focal_alpha if args.focal_alpha is not None else min(ratio / (1 + ratio), 0.99)
        alpha_o = 1.0 - alpha_b
        alpha_tensor = torch.tensor([alpha_o, alpha_b], dtype=torch.float32, device=device)
        loss_fn = FocalLoss(alpha=alpha_tensor, gamma=args.focal_gamma)
        print(f"  Focal Loss: gamma={args.focal_gamma}, alpha=[O={alpha_o:.4f}, B={alpha_b:.4f}]")
    else:
        pw = args.pos_weight if args.pos_weight is not None else min(math.sqrt(ratio), 500.0)
        class_weights = torch.tensor([1.0, pw], dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        print(f"  Weighted CE: pos_weight={pw:.1f}")

    # ── DataLoaders ──
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.workers,
        pin_memory=pin, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.workers,
        pin_memory=pin, persistent_workers=args.workers > 0,
    )

    # ── Optimiser & schedule ──
    raw_model = model.module if isinstance(model, (nn.DataParallel, torch._dynamo.eval_frame.OptimizedModule)) else model
    if hasattr(raw_model, '_orig_mod'):
        raw_model = raw_model._orig_mod  # unwrap torch.compile

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    steps_per_epoch = len(train_loader) // args.grad_accumulation
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
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
    if args.cost_per_hour:
        print(f"  Cost tracking:        ${args.cost_per_hour:.2f}/hr")

    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    global_step = 0
    training_log = []
    train_start = time.time()

    print("\n" + "=" * 70)
    print("Starting training")
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

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
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
                eta_epoch = (len(train_loader) - step - 1) / max(it_per_sec, 0.01)

                postfix = {
                    "loss": f"{loss.item() * args.grad_accumulation:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.1e}",
                    "it/s": f"{it_per_sec:.1f}",
                }
                if args.cost_per_hour:
                    postfix["$"] = f"{elapsed / 3600 * args.cost_per_hour:.2f}"
                pbar.set_postfix(postfix)

                # ── Eval ──
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

                    cost_str = f"  ${log_entry.get('cost_usd', 0):.2f}" if args.cost_per_hour else ""
                    print(
                        f"\n  Step {global_step} [{fmt_time(elapsed)}{cost_str}]: "
                        f"val_loss={metrics['loss']:.4f}, "
                        f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
                        f"F1={metrics['f1']:.3f} "
                        f"(tp={metrics['tp']}, fp={metrics['fp']}, fn={metrics['fn']})"
                    )

                    if metrics["f1"] > best_f1:
                        best_f1 = metrics["f1"]
                        best_dir = checkpoints_dir / "best"
                        save_model = raw_model
                        save_model.save_pretrained(str(best_dir))
                        tokenizer.save_pretrained(str(best_dir))
                        print(f"  >>> New best F1={best_f1:.4f} — saved to {best_dir}")

                    model.train()

                # ── Checkpoint ──
                if global_step % args.save_steps == 0:
                    ckpt_dir = checkpoints_dir / f"checkpoint-{global_step}"
                    save_model = raw_model
                    save_model.save_pretrained(str(ckpt_dir))
                    tokenizer.save_pretrained(str(ckpt_dir))

        # ── End of epoch ──
        epoch_elapsed = time.time() - epoch_start
        avg_loss = epoch_loss / len(train_loader)
        total_elapsed = time.time() - train_start
        print(f"\nEpoch {epoch+1} done in {fmt_time(epoch_elapsed)}  |  avg loss: {avg_loss:.4f}")

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

        cost_str = f"  ${log_entry.get('cost_usd', 0):.2f}" if args.cost_per_hour else ""
        print(
            f"  Val: loss={metrics['loss']:.4f}, "
            f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
            f"F1={metrics['f1']:.3f}{cost_str}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_dir = checkpoints_dir / "best"
            save_model = raw_model
            save_model.save_pretrained(str(best_dir))
            tokenizer.save_pretrained(str(best_dir))
            print(f"  >>> New best F1={best_f1:.4f}")

    # ── Final save ──
    final_dir = checkpoints_dir / "final"
    save_model = raw_model
    save_model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    total_time = time.time() - train_start
    with open(args.output_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Training complete in {fmt_time(total_time)}")
    if args.cost_per_hour:
        print(f"Estimated cost: ${total_time / 3600 * args.cost_per_hour:.2f}")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Best model:  {checkpoints_dir / 'best'}")
    print(f"Final model: {final_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
