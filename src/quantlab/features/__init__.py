"""Feature engineering with explicit temporal guarantees."""

from quantlab.features.base import FeatureSetSpec, FeatureSpec
from quantlab.features.guards import assert_causal, assert_split_ordering
from quantlab.features.library import default_feature_set
from quantlab.features.pipeline import (
    build_feature_frame,
    build_labels,
    build_training_frame,
)

__all__ = [
    "FeatureSetSpec",
    "FeatureSpec",
    "assert_causal",
    "assert_split_ordering",
    "build_feature_frame",
    "build_labels",
    "build_training_frame",
    "default_feature_set",
]
