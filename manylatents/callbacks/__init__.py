"""Callbacks for manylatents experiment pipeline."""
from manylatents.callbacks.callback import BaseCallback
from manylatents.callbacks.debug import FirstBatchLoggerCallback, attach_nan_detector
from manylatents.callbacks.embedding.base import EmbeddingCallback
from manylatents.callbacks.embedding.save_embeddings import SaveEmbeddings
from manylatents.callbacks.embedding.loadings_analysis import LoadingsAnalysisCallback

__all__ = [
    "BaseCallback",
    "EmbeddingCallback",
    "FirstBatchLoggerCallback",
    "LoadingsAnalysisCallback",
    "SaveEmbeddings",
    "attach_nan_detector",
]
