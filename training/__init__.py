"""Training routing and training components for the BAGEL backbone."""

from .config import DepthPlan, DynamicDepthConfig, DynamicDepthController
from .tafe import (
    AttentionResidualSubset,
    FFNResidualSubset,
    FFNSubsetPool,
    RoutedAttention,
    RoutedFFN,
    TAFEAction,
    TAFEExecutionState,
    TAFEGate,
    TAFEOutput,
    current_tafe_state,
    tafe_execution,
)

_LAZY_MODELING = {
    "DynamicBagel",
    "DynamicQwen2Model",
    "enable_dynamic_depth",
    "set_training_stage",
}
_LAZY_FUSION = {
    "FusionCompiler",
    "FusionRealizationAdapter",
    "FusionReleaseController",
}
_LAZY_ROUTER = {"CandidateDepthRouter"}

__all__ = [
    "CandidateDepthRouter",
    "AttentionResidualSubset",
    "DepthPlan",
    "DynamicBagel",
    "DynamicDepthConfig",
    "DynamicDepthController",
    "DynamicQwen2Model",
    "FFNResidualSubset",
    "FFNSubsetPool",
    "FusionCompiler",
    "FusionRealizationAdapter",
    "FusionReleaseController",
    "RoutedFFN",
    "RoutedAttention",
    "TAFEAction",
    "TAFEExecutionState",
    "TAFEGate",
    "TAFEOutput",
    "current_tafe_state",
    "enable_dynamic_depth",
    "set_training_stage",
    "tafe_execution",
]


def __getattr__(name: str):
    if name in _LAZY_MODELING:
        from . import modeling

        return getattr(modeling, name)
    if name in _LAZY_FUSION:
        from . import fusion

        return getattr(fusion, name)
    if name in _LAZY_ROUTER:
        from . import router

        return getattr(router, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
