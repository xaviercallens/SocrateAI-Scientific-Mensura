"""SOCRATES: unified algebraic-topological framework for multi-scale scientific data.

The same computational lens -- polynomial algebra plus persistent homology --
applied across scales from Calabi-Yau compactifications to the cosmic web.
"""

__version__ = "0.1.0"

from . import core, tda

__all__ = ["core", "tda", "__version__"]
