"""Utilities for approximating Hessian matrices.

References:
    1. Schlegel, Theor. Chim. Acta, 66, 333 (1984)
"""
import typing

import torch

import tico.ic

if typing.TYPE_CHECKING:
    import tico.opt


def _guess_hess_distance(
    coords: torch.Tensor, atomic_nums: torch.Tensor, idxs: torch.Tensor
) -> list[float]:
    dist = torch.linalg.norm(coords[idxs[:, 0]] - coords[idxs[:, 1]], dim=1)
    hess = []

    for (idx_a, idx_b), r in zip(idxs, dist, strict=True):
        a = 1.734

        z_1 = min(atomic_nums[idx_a], atomic_nums[idx_b])
        z_2 = max(atomic_nums[idx_a], atomic_nums[idx_b])

        if z_1 < 3:
            if z_2 < 3:
                b = -0.244
            elif z_2 < 11:
                b = 0.352
            else:
                b = 0.660
        elif z_1 < 11:
            if z_2 < 11:
                b = 1.085
            else:
                b = 1.522
        else:
            b = 2.068

        hess.append((a / (r - b) ** 3).detach().cpu().item())

    return hess


def _guess_hess_angle(
    atomic_nums: torch.Tensor,
    idxs: torch.Tensor,
    is_bound: typing.Callable[[int, int], bool],
) -> list[float]:
    hess = []

    for idx_a, idx_b, idx_c in idxs:
        if min(atomic_nums[idx_a], atomic_nums[idx_b], atomic_nums[idx_c]) < 3:
            a = 0.160
        else:
            a = 0.250

        if is_bound(idx_a, idx_b) and is_bound(idx_b, idx_c):
            hess.append(a)
        else:
            hess.append(0.1)

    return hess


def _guess_hess_out_of_plane(
    idxs: torch.Tensor, is_bound: typing.Callable[[int, int], bool]
) -> list[float]:
    hess = []

    for idx_a, idx_b, idx_c, idx_d in idxs:
        if is_bound(idx_a, idx_b) and is_bound(idx_a, idx_c) and is_bound(idx_a, idx_d):
            hess.append(0.045)
        else:
            hess.append(0.023)

    return hess


def guess_hess_q(
    coords: torch.Tensor, ic_idxs: tico.ic.ICDict, atomic_nums: torch.Tensor
) -> torch.Tensor:
    """Build a guess Hessian that roughly follows Schlegel's guidelines."""

    hess_diag = []

    bonds = {
        tuple(sorted([int(idx_a), int(idx_b)]))
        for idx_a, idx_b in ic_idxs[tico.ic.ICType.DISTANCE]
    }

    def is_bound(idx_a, idx_b):
        return tuple(sorted([int(idx_a), int(idx_b)])) in bonds

    for ic_type, idxs in ic_idxs.items():
        if ic_type == tico.ic.ICType.DISTANCE:
            hess_diag.extend(_guess_hess_distance(coords, atomic_nums, idxs))
        elif ic_type in {tico.ic.ICType.ANGLE, tico.ic.ICType.LINEAR}:
            hess_diag.extend(_guess_hess_angle(atomic_nums, idxs, is_bound))
        elif ic_type == tico.ic.ICType.DIHEDRAL:
            hess_diag.extend([0.023] * len(idxs))
        elif ic_type == tico.ic.ICType.OUT_OF_PLANE:
            hess_diag.extend(_guess_hess_out_of_plane(idxs, is_bound))
        else:
            raise NotImplementedError()

    return torch.diag(torch.tensor(hess_diag, dtype=coords.dtype, device=coords.device))


def update_hess_q(
    hess_q: torch.Tensor,
    history: list["tico.opt.Step"],
    ic: tico.ic.IC,
    max_updates: int = 1,
) -> torch.Tensor:
    if len(history) < 2:
        return hess_q

    n_steps = 0

    for _ in range(2, len(history) + 1):
        if n_steps == max_updates:
            break
        n_steps += 1

    for i in range(n_steps):
        this_step = -n_steps + i
        prev_step = -n_steps + i - 1

        dq = ic.compute_dq(
            history[this_step].coords_x, history[prev_step].coords_x
        ).unsqueeze(-1)
        dg = (history[prev_step].grad_q - history[this_step].grad_q).unsqueeze(-1)

        if torch.linalg.norm(dq) < 1e-6:
            continue
        if torch.linalg.norm(dg) < 1e-6:
            continue

        mat_1 = (dg @ dg.T) / (dg.T @ dq)
        mat_2 = ((hess_q @ dq) @ (hess_q @ dq).T) / torch.linalg.multi_dot(
            [dq.T, hess_q, dq]
        )
        hess_q = hess_q + mat_1 - mat_2

    return hess_q
