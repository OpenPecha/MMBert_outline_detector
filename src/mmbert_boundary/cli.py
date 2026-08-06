"""Unified CLI entry point for the mmbert_boundary package.

Dispatches to per-stage ``main()`` functions based on the chosen subcommand.

Usage:
    mmbert-boundary prepare-data   [options]
    mmbert-boundary sample-benchmark [options]
    mmbert-boundary train          [options]
    mmbert-boundary evaluate       [options]
    mmbert-boundary predict <input> [options]
"""

from __future__ import annotations

import sys

_SUBCOMMANDS = ("prepare-data", "sample-benchmark", "train", "evaluate", "predict")

_HELP = """\
usage: mmbert-boundary <subcommand> [options]

mmBERT Tibetan text boundary detection pipeline

Subcommands
-----------
  prepare-data      Tokenise annotated volumes into a HuggingFace DatasetDict
  sample-benchmark  Build a stratified benchmark subset from the test split
  train             Fine-tune mmBERT on the prepared dataset
  evaluate          Evaluate a trained model against ground-truth boundaries
  predict           Run inference on raw .txt files

Run `mmbert-boundary <subcommand> --help` for per-subcommand options.
"""


def main(argv: list[str] | None = None) -> None:
    """Top-level dispatcher for all mmbert-boundary subcommands.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # Show top-level help when called with no args or an explicit help flag.
    if not argv or argv[0] in ("-h", "--help"):
        print(_HELP)
        return

    subcommand = argv[0]
    remaining = argv[1:]

    if subcommand == "prepare-data":
        from mmbert_boundary.core.prepare import main as _main
        _main(remaining)

    elif subcommand == "sample-benchmark":
        from mmbert_boundary.core.benchmark import main as _main
        _main(remaining)

    elif subcommand == "train":
        from mmbert_boundary.core.train import main as _main
        _main(remaining)

    elif subcommand == "evaluate":
        from mmbert_boundary.core.evaluate import main as _main
        _main(remaining)

    elif subcommand == "predict":
        from mmbert_boundary.core.predict import main as _main
        _main(remaining)

    else:
        print(f"mmbert-boundary: unknown subcommand '{subcommand}'\n", file=sys.stderr)
        print(_HELP)
        sys.exit(1)
