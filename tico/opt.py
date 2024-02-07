"""Optimize molecule geometrics in internal coordinates.

Notes:
    * This module is heavily inspired off of the ``optimize`` and ``step`` modules of
      ``geomeTRIC``. See the LICENSE-3RD-PARTY for license information.
"""
import dataclasses
import functools
import logging
import math
import random
import typing

import torch
import torch.autograd.functional

import tico.hess
import tico.ic
import tico.utils

EnergyFn = typing.Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


class Step(typing.NamedTuple):
    """The outputs of step in the optimization process."""

    coords_x: torch.Tensor
    """The cartesian coordinates [a0]."""
    grad_x: torch.Tensor
    """The gradients [Eh/a0] in cartesian coordinates."""

    coords_q: torch.Tensor
    """The internal coordinates."""
    grad_q: torch.Tensor
    """The gradients in internal coordinates."""


class ConvergenceCriteria(typing.TypedDict):
    """The convergence criteria for the optimization process."""

    energy: float
    """The energy [Eh] convergence criteria."""
    rms_grad: float
    """The root mean square of the gradients [Eh/a0] convergence criteria."""
    max_grad: float
    """The maximum gradient [Eh/a0] convergence criteria."""
    rms_disp: float
    """The root mean square of the displacements [a0] convergence criteria."""
    max_disp: float
    """The maximum displacement [a0] convergence criteria."""


_LOGGER = logging.getLogger(__name__)


_BOHR_TO_ANGSTROM = 0.529177210903
_ANGSTROM_TO_BOHR = 1 / _BOHR_TO_ANGSTROM


CONVERGENCE_CRITERIA: dict[str, ConvergenceCriteria] = {
    "GAU": {
        "energy": 1e-6,
        "rms_grad": 3e-4,
        "max_grad": 4.5e-4,
        "rms_disp": 1.2e-3 * _ANGSTROM_TO_BOHR,
        "max_disp": 1.8e-3 * _ANGSTROM_TO_BOHR,
    },
    "GAU_LOOSE": {
        "energy": 1e-6,
        "rms_grad": 1.7e-3,
        "max_grad": 2.5e-3,
        "rms_disp": 6.7e-3 * _ANGSTROM_TO_BOHR,
        "max_disp": 1e-2 * _ANGSTROM_TO_BOHR,
    },
    "GAU_TIGHT": {
        "energy": 1e-6,
        "rms_grad": 1e-5,
        "max_grad": 1.5e-5,
        "rms_disp": 4e-5 * _ANGSTROM_TO_BOHR,
        "max_disp": 6e-5 * _ANGSTROM_TO_BOHR,
    },
}
"""Default convergence criteria for optimization."""


@dataclasses.dataclass
class Params:
    """Parameters for the optimization process."""

    max_steps: int = 300
    """The maximum number of steps to take."""

    epsilon: float = 1e-5
    """The minimum eigenvalue of the Hessian."""

    trust: float = 0.1 * _ANGSTROM_TO_BOHR
    """The initial trust radius [a0]. This will be adjusted during the optimization
    process."""
    trust_min: float = 1.2e-3 * _ANGSTROM_TO_BOHR
    """The lower bound of the trust radius [a0]."""
    trust_max: float = 0.3 * _ANGSTROM_TO_BOHR
    """The upper bound of the trust radius [a0]."""

    criteria: ConvergenceCriteria = dataclasses.field(
        default_factory=lambda: {**CONVERGENCE_CRITERIA["GAU"]}
    )
    """The convergence criteria to use."""


def _compute_grad_norm(
    coords_x: torch.Tensor, grad_x: torch.Tensor, ic: tico.ic.IC
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the norm of the gradient, taking into account constraints."""
    if isinstance(ic, tico.ic.DLC) and ic.constr is not None:
        grad_x = ic.project_grad_x(coords_x, grad_x)

    grad_norm = torch.sqrt(torch.sum((grad_x.reshape(-1, 3)) ** 2, dim=1))

    rms_grad = torch.sqrt(torch.mean(grad_norm**2))
    max_grad = torch.max(grad_norm)

    return rms_grad, max_grad


def _compute_grad_q(
    coords_x: torch.Tensor, grad_x: torch.Tensor, ic: tico.ic.IC
) -> torch.Tensor:
    """Project the gradient into internal coordinates."""
    b_matrix = ic.compute_b(coords_x)
    g_matrix_inv = tico.utils.pinv(b_matrix @ b_matrix.T)

    return torch.linalg.multi_dot([g_matrix_inv, b_matrix, grad_x])


def _compute_dq(
    step_size,
    coords_x: torch.Tensor,
    grad_q: torch.Tensor,
    hess_q: torch.Tensor,
    ic: tico.ic.IC,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the delta in internal coordinates to take."""
    grad_q_len = len(grad_q)

    if isinstance(ic, tico.ic.DLC) and ic.constr is not None:
        grad_q, hess_q = ic.augment_hess_q(coords_x, grad_q, hess_q)

    hess_q = hess_q + step_size * torch.eye(len(hess_q))

    for i in range(grad_q_len, len(grad_q)):
        # don't shift the constrained degrees of freedom
        hess_q[i, i] = 0.0

    hess_q_inv = tico.utils.pinv(hess_q, 1e-12)

    dq_with_const = -hess_q_inv @ grad_q
    dq = dq_with_const[:grad_q_len]

    d_prime = -hess_q_inv @ dq_with_const
    dq_prime = dq @ d_prime[:grad_q_len] / torch.linalg.norm(dq)

    return dq, dq_prime


def _find_dq(
    target_norm: float,
    step_size: float,
    coords_x: torch.Tensor,
    grad_q: torch.Tensor,
    hess_q: torch.Tensor,
    ic: tico.ic.IC,
    rtol: float = 0.001,
) -> torch.Tensor:
    """

    Notes:
        Based on geomeTRIC, which is based on the iterative formula from Hebden (1973),
        equation 5.2.10 in "Practical methods of optimization" by Fletcher

    Returns:
        The found value of dq.
    """
    dq, dq_prime = _compute_dq(step_size, coords_x, grad_q, hess_q, ic)
    dq_norm = torch.linalg.norm(dq)

    if dq_norm < target_norm:
        return dq

    dq_norm_last, dq_norm_best, dq_best = 0.0, dq_norm, dq.clone()

    iteration = 0

    while True:
        step_size += (1 - dq_norm / target_norm) * (dq_norm / dq_prime)

        dq, dq_prime = _compute_dq(step_size, coords_x, grad_q, hess_q, ic)
        dq_norm = torch.linalg.norm(dq)

        if torch.abs((dq_norm - target_norm) / target_norm) < rtol:
            return dq
        if iteration > 10 and torch.abs(dq_norm_last - dq_norm) / dq_norm < rtol:
            return dq

        iteration += 1

        dq_norm_last = dq_norm

        if dq_norm < dq_norm_best:
            dq_norm_best, dq_best = dq_norm, dq.clone()

        # handle infinite loops... following geomeTRIC
        if iteration % 100 == 99:
            step_size += random.random() * iteration / 100
        if iteration % 1000 == 999:
            return dq_best


def _line_search(
    trust: float,
    step_size: float,
    coords_x: torch.Tensor,
    grad_q: torch.Tensor,
    hess_q: torch.Tensor,
    dq: torch.Tensor,
    ic: tico.ic.IC,
    max_trials: int = 4,
) -> torch.Tensor:
    """Perform a line search to find a step size that matches the trust radius."""
    assert max_trials >= 1

    @functools.lru_cache
    def evaluate_dx(trial: float) -> tuple[torch.Tensor, bool]:
        dq_trial = _find_dq(trial, step_size, coords_x, grad_q, hess_q, ic)

        coords_x_new, _, q_converged = ic.dq_to_x(coords_x, dq_trial)
        dx = tico.utils.compute_rmsd(coords_x_new, coords_x)[0]

        return dx, q_converged

    def evaluate(trial: float, target_dx: float) -> tuple[float, bool]:
        if trial == 0.0:
            return -trust, False

        dx, q_converged = evaluate_dx(trial)
        return (dx - target_dx), q_converged

    dq_norm = torch.linalg.norm(dq).detach().cpu().item()

    is_trial_valid = True  # could we map dq to new cartesian coordinates?
    target = trust

    for i in range(max_trials):
        dq_norm, trials, converged = tico.utils.brent(
            evaluate, 0.0, dq_norm, target, cvg=0.1, args=(target,)
        )

        best_trials = sorted(
            ((trial, delta) for trial, delta, _ in trials if delta < 0),
            key=lambda x: x[1],
        )
        is_trial_valid = trials[-1][-1]

        if i == 0:
            if not converged and len(best_trials) > 0:
                dq_norm = best_trials[0][0]
                break
            elif not is_trial_valid:
                pass
            else:
                break
        elif is_trial_valid:
            break

        target *= 0.5

    if not is_trial_valid:  # could not find a valid step that lets x converge...
        raise RuntimeError("line search failed to find a valid step")

    return _find_dq(dq_norm, step_size, coords_x, grad_q, hess_q, ic)


def _update_trust(
    step_quality: float,
    rms_disp: torch.Tensor,
    converged_energy: bool,
    trust: float,
    trust_min: float,
    trust_max: float,
    constr_violation: bool,
):
    prev_trust = trust

    if step_quality > 0.75:
        trust = min(trust_max, math.sqrt(2) * trust)
    elif step_quality <= 0.25:
        trust = max(trust_min, min(trust, rms_disp.detach().cpu().item()) / 2)
        # TODO: check for large translation/rotations

    reject_step = step_quality < -1.0 and not (
        prev_trust <= trust_min
        or rms_disp <= 1.2 * trust_max
        or step_quality < 0.0
        or converged_energy
        or constr_violation
    )
    return trust, reject_step


def _has_converged(
    energy: torch.Tensor,
    energy_new: torch.Tensor,
    rms_grad: torch.Tensor,
    max_grad: torch.Tensor,
    rms_disp: torch.Tensor,
    max_disp: torch.Tensor,
    params: Params,
) -> bool:
    converged_energy = torch.abs(energy_new - energy) < params.criteria["energy"]

    _LOGGER.info(
        f"Eold={energy: .4f} "
        f"Enew={energy_new: .4f} "
        f"ΔE={float(energy_new - energy): .4f} "
        f"rms_grad={float(rms_grad): .4f} "
        f"max_grad={float(max_grad): .4f} "
        f"rms_disp={float(rms_disp): .4f} "
        f"max_disp={float(max_disp): .4f}"
    )

    return (
        converged_energy
        and rms_grad < params.criteria["rms_grad"]
        and max_grad < params.criteria["max_grad"]
        and rms_disp < params.criteria["rms_disp"]
        and max_disp < params.criteria["max_disp"]
    )


def optimize(
    coords_x: torch.Tensor,
    ic: tico.ic.IC,
    energy_fn: EnergyFn,
    atomic_nums: torch.Tensor,
    params: Params | None = None,
) -> tuple[list[Step], bool]:
    """Optimize the geometry of a molecule.

    Args:
        coords_x: The initial cartesian coordinates [a0].
        ic: The internal coordinate representation of the molecule.
        energy_fn: A function that computes the energy and gradients of the molecule.
            It should take the cartesian coordinates and return the energy [Eh] and
            gradients [Eh/a0] in atomic units.
        atomic_nums: The atomic numbers of the atoms in the molecule.
        params: The parameters for the optimization.

    Returns:
        The history of the optimization process and whether the optimization converged
        or not.
    """
    params = params if params is not None else Params()

    trust = params.trust
    history = []

    reject_step = False

    coords_x = coords_x.double()
    coords_q = ic.compute_q(coords_x)

    hess_q = ic.guess_hess_q(coords_x, atomic_nums)
    energy, grad_x = energy_fn(coords_x)

    for _ in range(params.max_steps):
        grad_q = _compute_grad_q(coords_x, grad_x, ic)

        if not reject_step:
            history.append(Step(coords_x, grad_x, coords_q, grad_q))
            hess_q = tico.hess.update_hess_q(hess_q, history, ic)

        min_eig_val = sorted(torch.linalg.eigh(hess_q)[0])[0].real
        step_size = (
            params.epsilon - min_eig_val if min_eig_val < params.epsilon else 0.0
        )

        dq, _ = _compute_dq(step_size, coords_x, grad_q, hess_q, ic)
        coords_x_new, coords_q_new, q_converged = ic.dq_to_x(coords_x, dq)

        if tico.utils.compute_rmsd(coords_x_new, coords_x)[0] > 1.1 * trust:
            dq = _line_search(trust, step_size, coords_x, grad_q, hess_q, dq, ic)
            coords_x_new, coords_q_new, q_converged = ic.dq_to_x(coords_x, dq)

        energy_new, grad_x_new = energy_fn(coords_x_new)

        rms_grad, max_grad = _compute_grad_norm(coords_x_new, grad_x_new, ic)
        rms_disp, max_disp = tico.utils.compute_rmsd(coords_x_new, coords_x)

        if _has_converged(
            energy, energy_new, rms_grad, max_grad, rms_disp, max_disp, params
        ):
            grad_q_new = _compute_grad_q(coords_x, grad_x, ic)

            history = history + [
                Step(coords_x_new, grad_x_new, coords_q_new, grad_q_new)
            ]
            return history, True

        expected_improve = float(
            0.5 * torch.linalg.multi_dot([dq.unsqueeze(0), hess_q, dq])
            + torch.dot(dq, grad_q)
        )
        step_quality = (energy_new - energy) / expected_improve

        constr_violation = (
            isinstance(ic, tico.ic.DLC)
            and ic.constr is not None
            and torch.max(torch.abs(ic.compute_constr_delta(coords_x_new))) > 1e-1
        )

        trust, reject_step = _update_trust(
            step_quality,
            rms_disp,
            torch.abs(energy_new - energy) < params.criteria["energy"],
            trust,
            params.trust_min,
            params.trust_max,
            constr_violation,
        )

        if not reject_step:
            coords_x, coords_q = coords_x_new, coords_q_new
            energy, grad_x = energy_new, grad_x_new

    return history, False
