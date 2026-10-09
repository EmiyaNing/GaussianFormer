"""P1/P3 ablation adapters (controls for the Gaussian necessity question)."""
from .knn_residual import KNNResidualAggregator
from .dense_g2v_residual import DenseG2VResidualAggregator

__all__ = ['KNNResidualAggregator', 'DenseG2VResidualAggregator']
