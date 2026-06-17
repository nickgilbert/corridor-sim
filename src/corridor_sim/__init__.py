"""Corridor Economics Simulation Engine.

A Monte Carlo engine that quantifies the all-in cross-currency spread of the
denomination-matched reserved-ramp value chain (Model C) against two alternative
ramp/FX architectures (Models A and B), per the V6 strategy specification.
"""

__version__ = "1.0.0"

from .config import Config, load_config, build_default_config  # noqa: F401
from .economics import spread_components, yield_subsidy_bps  # noqa: F401
from .models import MODELS, ModelSpec  # noqa: F401
