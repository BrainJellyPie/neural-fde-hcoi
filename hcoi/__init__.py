"""History-compatible identification of the fractional order in Neural FDEs.

Reference implementation accompanying the paper
"Identifiability and Deterministic Stable Recovery of the Fractional Order in
Neural Fractional Differential Equations".
"""

from . import caputo, diagnostics, identification, systems

__all__ = ["caputo", "diagnostics", "identification", "systems"]
__version__ = "1.0.0"
