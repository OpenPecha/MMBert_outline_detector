"""mmbert_boundary — Tibetan text boundary detection with mmBERT."""

__version__ = "0.1.0"

from mmbert_boundary.core.predict import annotate_text, predict_boundaries

__all__ = ["predict_boundaries", "annotate_text", "__version__"]
