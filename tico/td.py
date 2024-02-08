"""Run torsion drives."""
import collections
import dataclasses
import logging
import typing

import smee.geometry
import torch

import tico.ic
import tico.opt

if typing.TYPE_CHECKING:
    import openff.toolkit

_LOGGER = logging.getLogger(__name__)


_ELEMENT_LOOKUP = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}


@dataclasses.dataclass
class WFP:
    """Parameters for torsion drives that use wavefront propagation."""

    type: typing.Literal["wavefront"] = "wavefront"
    """The type of driver to use."""

    energy_decrease_thresh: float | None = None
    """The minimum energy decrease required to continue scanning."""
    energy_upper_limit: float | None = None
    """The maximum energy to scan up to."""


@dataclasses.dataclass
class Simple:
    """Parameters for torsion drives that do a simple linear scan."""

    type: typing.Literal["simple"] = "simple"
    """The type of driver to use."""


@dataclasses.dataclass
class Params:
    """Parameters for the optimization process."""

    driver: WFP | Simple = dataclasses.field(default_factory=Simple)
    """The algorithm to use for the torsion drive."""

    grid_spacing: int = 15
    """The spacing [deg] between each grid point."""

    opt: tico.opt.Params = dataclasses.field(default_factory=tico.opt.Params)
    """The parameters to use for the optimization at each angle."""


def bond_to_dihedral_idxs(
    mol: "openff.toolkit.Molecule", bond: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Select a dihedral to scan given a central bond to scan around.

    Args:
        mol: The molecule to scan.
        bond: The bond to scan around.

    Returns:
        The indices of the atoms involved in the dihedral.
    """
    idx_2, idx_3 = bond

    atoms_1 = [
        atom
        for atom in mol.atoms[idx_2].bonded_atoms
        if atom.molecule_atom_index != idx_3
    ]
    idx_1 = max(atoms_1, key=lambda atom: atom.atomic_number).molecule_atom_index

    atoms_4 = [
        atom
        for atom in mol.atoms[idx_3].bonded_atoms
        if atom.molecule_atom_index != idx_2
    ]
    idx_4 = max(atoms_4, key=lambda atom: atom.atomic_number).molecule_atom_index

    return idx_1, idx_2, idx_3, idx_4


def _torsion_drive_wavefront(
    coords: torch.Tensor | list[torch.Tensor],
    bond_idxs: torch.Tensor,
    dihedral: tuple[int, int, int, int],
    energy_fn: tico.opt.EnergyFn,
    atomic_nums: torch.Tensor,
    params: Params | None = None,
) -> dict[int, list[tuple[torch.Tensor, torch.Tensor]]]:
    import torsiondrive.td_api

    coords = [coords] if not isinstance(coords, list) else coords

    state = torsiondrive.td_api.create_initial_state(
        dihedrals=[dihedral],
        grid_spacing=[params.grid_spacing],
        elements=[_ELEMENT_LOOKUP[int(atomic_num)] for atomic_num in atomic_nums],
        init_coords=[coord.flatten().tolist() for coord in coords],
        dihedral_ranges=None,
        energy_upper_limit=params.driver.energy_upper_limit,
        energy_decrease_thresh=params.driver.energy_decrease_thresh,
    )

    results = collections.defaultdict(list)
    counter = 0

    while True:
        next_jobs = torsiondrive.td_api.next_jobs_from_state(state, verbose=False)

        if len(next_jobs) == 0:
            break

        angle_results = {}

        for angle, jobs in next_jobs.items():
            constr = {
                tico.ic.ICType.DIHEDRAL: (
                    torch.tensor([dihedral]),
                    torch.tensor([float(angle) * torch.pi / 180.0]),
                )
            }

            _LOGGER.info(
                f"optimizing angle {angle}... N finished={counter} N jobs={len(jobs)}"
            )

            job_results = []

            for coords_start_flat in jobs:
                coords_start = torch.tensor(
                    coords_start_flat, dtype=torch.float64
                ).reshape(-1, 3)

                ic = tico.ic.DLC.from_coords(coords_start, bond_idxs, constr)

                steps, converged = tico.opt.optimize(
                    coords_start, ic, energy_fn, atomic_nums, params.opt
                )
                assert converged

                coords_final = steps[-1].coords_x
                energy_final, _ = energy_fn(coords_final)

                coords_final_flat = coords_final.flatten().tolist()

                job_results.append((coords_start_flat, coords_final_flat, energy_final))

                counter += 1

            angle_results[angle] = job_results
            results[angle].extend(job_results)

        torsiondrive.td_api.update_state(state, {**angle_results})

    results = {
        int(angle): [
            (torch.tensor(coords_final_flat).reshape(-1, 3), energy_final)
            for (_, coords_final_flat, energy_final) in results[angle]
        ]
        for angle in results
    }
    return results


def _torsion_drive_simple(
    coords: torch.Tensor | list[torch.Tensor],
    bond_idxs: torch.Tensor,
    dihedral: tuple[int, int, int, int],
    energy_fn: tico.opt.EnergyFn,
    atomic_nums: torch.Tensor,
    params: Params | None = None,
) -> dict[int, list[tuple[torch.Tensor, torch.Tensor]]]:
    """Optimize at each angle in a torsion drive sequentially."""

    if not isinstance(coords, torch.Tensor):
        raise NotImplementedError(
            "simple torsion drives only support a single initial coordinate"
        )

    dihedral = torch.tensor([dihedral])
    angle_0 = smee.geometry.compute_dihedrals(coords, dihedral)

    angles = torch.tensor(
        range(
            -180 + params.grid_spacing, 180 + params.grid_spacing, params.grid_spacing
        ),
        dtype=torch.int64,
    )

    start_idx = (angles - torch.rad2deg(angle_0)).abs().argmin()

    angles = torch.cat([angles[start_idx:], angles[:start_idx]])
    results = {}

    for i, angle in enumerate(angles):
        angle_rad = torch.deg2rad(angle.double()).unsqueeze(0)
        constr = {tico.ic.ICType.DIHEDRAL: (dihedral, angle_rad)}

        ic = tico.ic.DLC.from_coords(coords, bond_idxs, constr)

        _LOGGER.info(f"optimizing angle {angle} [{i+1}/{len(angles)}]")

        steps, converged = tico.opt.optimize(
            coords, ic, energy_fn, atomic_nums, params.opt
        )
        assert converged

        coords = steps[-1].coords_x
        energy, _ = energy_fn(coords)

        results[int(angle)] = [(coords, energy)]

    return results


def torsion_drive(
    coords: torch.Tensor | list[torch.Tensor],
    bond_idxs: torch.Tensor,
    dihedral: tuple[int, int, int, int],
    energy_fn: tico.opt.EnergyFn,
    atomic_nums: torch.Tensor,
    params: Params | None = None,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Perform a torsion scan of a molecule around a given dihedral.

    Args:
        coords: The initial cartesian coordinates [a0].
        bond_idxs: The indices of the bonds in the molecule to scan.
        dihedral: The dihedral to scan.
        energy_fn: A function that computes the energy and gradients of the molecule.
            It should take the cartesian coordinates and return the energy [Eh] and
            gradients [Eh/a0] in atomic units.
        atomic_nums: The atomic numbers of the atoms in the molecule.
        params: The parameters for the torsion drive.

    Returns:
        The optimized coordinates [a0] and energies [Eh] at each angle [deg].
    """
    params = params if params is not None else Params()

    if params.driver.type.lower() == "wavefront":
        results = _torsion_drive_wavefront(
            coords, bond_idxs, dihedral, energy_fn, atomic_nums, params
        )
    elif params.driver.type.lower() == "simple":
        results = _torsion_drive_simple(
            coords, bond_idxs, dihedral, energy_fn, atomic_nums, params
        )
    else:
        raise NotImplementedError(f"{params.driver.type} is not supported")

    samples = {}

    for angle in results:
        final_energies = torch.tensor([energy for _, energy in results[angle]])

        lowest_energy_idx = final_energies.argmin()
        lowest_energy_result = results[angle][lowest_energy_idx]

        coords = lowest_energy_result[0].reshape(-1, 3)
        energy = lowest_energy_result[1]

        samples[angle] = coords, energy

    return samples
