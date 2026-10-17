"""Shot counting for mask manufacturability evaluation.

- `FastShotCounter`: the fast (>130x) maximal-rectangle + ILP cover estimator
  introduced in LithoGRPO.
- `ShotCounter`: the slower adabox-decomposition reference (LithoBench baseline).
"""

from .fast import FastShotCounter

__all__ = ["FastShotCounter", "ShotCounter"]


def __getattr__(name):
    # Reference-only dependencies (OpenCV/pandas) should not block fast counting.
    if name == 'ShotCounter':
        from .reference import ShotCounter
        return ShotCounter
    raise AttributeError(name)
