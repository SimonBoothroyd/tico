import torch
from scipy.stats import special_ortho_group

from tico.utils import brent, compute_rmsd, pinv


def test_pinv():
    matrix = torch.randn((5, 5)).double()
    matrix = matrix @ matrix.T

    matrix_pinv = pinv(matrix)

    result = matrix @ matrix_pinv
    expected = torch.eye(5, dtype=matrix.dtype)

    assert torch.allclose(result, expected, atol=1e-6)


def test_compute_rmsd():
    x = torch.randn((10, 3))

    rotation_matrix = torch.tensor(special_ortho_group.rvs(3), dtype=torch.float32)
    x_rotated = x @ rotation_matrix

    rms_disp, max_disp = compute_rmsd(x, x_rotated)

    assert isinstance(rms_disp, torch.Tensor)
    assert isinstance(max_disp, torch.Tensor)

    assert rms_disp.shape == torch.Size([])
    assert max_disp.shape == torch.Size([])

    assert torch.isclose(rms_disp, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(max_disp, torch.tensor(0.0), atol=1e-6)


def test_brent():
    def fn(x: float) -> tuple[float, bool]:
        return x - 2.0, True

    root, trials, converged = brent(fn, 1, 3, 1)

    assert abs(root - 2) < 1e-6
    assert converged
