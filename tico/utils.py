"""Common utility functions."""
import typing

import torch


def pinv(matrix: torch.Tensor, threshold: float = 1e-6) -> torch.Tensor:
    """Invert a matrix using SVD.

    Args:
        matrix: The matrix to invert.
        threshold: Eigenvalues below this value will be ignored.

    Returns:
        The inverted matrix.
    """
    return torch.linalg.pinv(matrix, rtol=threshold, hermitian=True)


def compute_rmsd(
    x: torch.Tensor, y: torch.Tensor, align: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the RMSD and maximum displacement between two sets of coordinates.

    Args:
        x: The first set of coordinates.
        y: The second set of coordinates.

    Returns:
        The RMSD and maximum displacement.
    """
    y = y.reshape(-1, 3) - torch.mean(y, dim=0)
    x = x.reshape(-1, 3) - torch.mean(x, dim=0)

    if align:
        covariance = x.T @ y

        u, _, vt = torch.linalg.svd(covariance)
        rotation = torch.matmul(vt.T, u.T)

        if torch.det(rotation) < 0:
            vt[-1, :] *= -1
            rotation = torch.matmul(vt.T, u.T)

        y = y @ rotation

    displacement = torch.sqrt(torch.sum((x - y).square(), dim=1))

    rms_disp = torch.sqrt(torch.mean(displacement**2))
    max_disp = torch.max(displacement)

    return rms_disp, max_disp


def brent(
    fn: typing.Callable[[float, typing.Any], tuple[float, bool]],
    a: float,
    b: float,
    rel: float,
    cvg: float = 0.1,
    args: tuple[typing.Any, ...] | None = None,
) -> tuple[float, list[tuple[float, float, bool]], bool]:
    """Find the root of a function using Brent's method.

    The algorithm is considered converged when ``abs(fs / rel) <= cvg``.

    Notes:
        Based on https://en.wikipedia.org/wiki/Brent (31/01/24).

    Args:
        fn: The function to evaluate.
        a: The minimum value of the starting bracket.
        b: The maximum value of the starting bracket.
        rel: The denominator used to calculate the fractional error.
        cvg: The convergence threshold for the relative error.
        args: Additional arguments to pass to the function.

    Returns:
        The found root, the attempted trials, and whether the algorithm converged.
    """
    args = () if args is None else args

    # taken to be constant with geomeTRIC
    delta, epsilon = 1e-6, min(0.01, 1e-2 * abs(a - b))

    fa, _ = fn(a, *args)
    fb, _ = fn(b, *args)

    if fa * fb >= 0:
        raise ValueError("values at the bracket endpoints must be of opposite sign")

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc, d = a, fa, None
    mflag = True

    trials = []

    while True:
        if fa != fc and fb != fc:
            s = a * fb * fc / ((fa - fb) * (fa - fc))
            s += b * fa * fc / ((fb - fa) * (fb - fc))
            s += c * fa * fb / ((fc - fa) * (fc - fb))
        else:
            s = b - fb * (b - a) / (fb - fa)

        bound = (3 * a + b) / 4

        condition_1 = not (min(b, bound) < s < max(b, bound))
        condition_2 = mflag and (abs(s - b) >= abs(b - c) / 2)
        condition_3 = (not mflag) and (abs(s - b) >= abs(c - d) / 2)
        condition_4 = mflag and (abs(b - c) < delta)
        condition_5 = (not mflag) and (abs(c - d) < delta)

        if condition_1 or condition_2 or condition_3 or condition_4 or condition_5:
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False

        fs, is_valid = fn(s, *args)
        trials.append((s, fs, is_valid))

        if abs(fs / rel) <= cvg:  # converged
            return s, trials, True
        if abs(b - a) < epsilon:  # failed - interval becomes too small
            return s, trials, False

        d, c = c, b
        fc = fb

        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
