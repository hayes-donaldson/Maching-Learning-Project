"""Data-driven wing surrogate. This package must not import solver."""

from .preprocessing import DatasetSplit, split_by_geometry

__all__ = ["DatasetSplit", "split_by_geometry"]
