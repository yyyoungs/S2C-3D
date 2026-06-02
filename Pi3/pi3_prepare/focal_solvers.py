import numpy as np


def solve_optimal_focal_shift(uv: np.ndarray, points: np.ndarray) -> tuple[float, float]:
    """Solve focal and z-shift from normalized image coordinates and point map."""
    xy = points[..., :2].reshape(-1, 2)
    z = points[..., 2].reshape(-1, 1)
    uv = uv.reshape(-1, 2)

    a = np.stack([xy, -uv], axis=-1).reshape(-1, 2)
    b = (uv * z).reshape(-1)
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    focal, shift = solution
    return float(shift), float(focal)


def solve_optimal_shift(uv: np.ndarray, points: np.ndarray, focal: float) -> float:
    """Solve z-shift when focal is already known."""
    xy = points[..., :2].reshape(-1, 2)
    z = points[..., 2].reshape(-1, 1)
    uv = uv.reshape(-1, 2)

    a = (-uv).reshape(-1, 1)
    b = (uv * z - focal * xy).reshape(-1)
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    return float(solution[0])
