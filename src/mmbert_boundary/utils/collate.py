"""DataLoader collate functions shared across train / evaluate."""

from typing import Any

import torch


def collate_train(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Pad a batch of windows to the same length for training.

    Does not include ``doc_ids`` — training does not need them.

    Args:
        batch: List of example dicts with ``input_ids``, ``attention_mask``,
            and ``labels`` (all Python lists of ints).

    Returns:
        Dict of stacked tensors: ``input_ids``, ``attention_mask``, ``labels``.
    """
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


def collate_eval(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad a batch of windows and preserve ``doc_ids`` for evaluation.

    Args:
        batch: List of example dicts that also contain a ``doc_id`` field.

    Returns:
        Dict with stacked tensors plus a plain list of ``doc_ids``.
    """
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids, attention_mask, labels, doc_ids = [], [], [], []
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
