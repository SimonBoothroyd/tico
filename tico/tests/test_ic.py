import collections
import math

import numpy
import openff.toolkit
import openff.units
import pytest
import torch

from tico.ic import RIC, ICType, _compute_linear_displacement, _compute_q


def _order_key(key: tuple[int, ...]) -> tuple[int, ...]:
    return key if key[-1] > key[0] else tuple(reversed(key))


def _compute_geometric_ic(
    molecule: openff.toolkit.Molecule, coords: torch.Tensor
) -> dict[ICType, tuple[torch.Tensor, torch.Tensor]]:
    """Computes the internal coordinate representation of a molecule using ``geomeTRIC``
    and returns the output in tensor form.

    Args:
        molecule: The molecule of interest.
        coords: The cartesian coordinates with shape=(n_atoms, 3).

    Returns:
        A dict of the internal coordinates with the form ``ic[ype] = (idxs, values)``.
        Angles will have units of [rad].
    """

    from geometric.internal import Angle, Dihedral, Distance, LinearAngle, OutOfPlane
    from geometric.internal import PrimitiveInternalCoordinates as GeometricPRIC
    from geometric.molecule import Molecule as GeometricMolecule

    geometric_molecule = GeometricMolecule()
    geometric_molecule.Data = {
        "resname": ["UNK"] * molecule.n_atoms,
        "resid": [0] * molecule.n_atoms,
        "elem": [atom.symbol for atom in molecule.atoms],
        "bonds": [(bond.atom1_index, bond.atom2_index) for bond in molecule.bonds],
        "name": molecule.name,
        "xyzs": [coords.detach().numpy()],
    }
    geometric_molecule.top_settings["read_bonds"] = True
    geometric_coords = GeometricPRIC(geometric_molecule)

    expected_coords = {
        Distance: (ICType.DISTANCE, ("a", "b")),
        Angle: (ICType.ANGLE, ("a", "b", "c")),
        LinearAngle: (ICType.LINEAR, ("a", "b", "c", "axis")),
        OutOfPlane: (ICType.OUT_OF_PLANE, ("a", "b", "c", "d")),
        Dihedral: (ICType.DIHEDRAL, ("a", "b", "c", "d")),
    }

    coords_by_type = collections.defaultdict(lambda: ([], []))

    for ic in geometric_coords.Internals:
        if type(ic) not in expected_coords:
            continue

        ic_name, idx_attributes = expected_coords[type(ic)]

        coords_by_type[ic_name][0].append(
            [getattr(ic, attr_name) for attr_name in idx_attributes]
        )
        coords_by_type[ic_name][1].append(ic.value(coords.detach().numpy()))

    return {
        key: (torch.tensor(idxs), torch.tensor(values))
        for key, (idxs, values) in coords_by_type.items()
    }


def _validate_ic(
    molecule: openff.toolkit.Molecule, coords: torch.Tensor, verbose: bool = False
):
    """Compares the values of the primitive redundant internal coordinates computed
    internally with those computed using ``geomeTRIC``.

    Args:
        molecule: The molecule of interest.
        coords: The coords of the molecule with shape=(n_atoms, 3).
        verbose: Whether to print information about any differences.

    Raises:
        AssertionError: If the internal coordinates do not match.
    """

    bond_idxs = torch.tensor(
        [[bond.atom1_index, bond.atom2_index] for bond in molecule.bonds]
    )

    actual_ic = RIC.from_coords(coords, bond_idxs).idxs
    actual_ic_q = _compute_q(coords, actual_ic)

    expected_ic = _compute_geometric_ic(molecule, coords)

    for ic_type in actual_ic:
        if verbose:
            print(str(ic_type).center(80, "="))

        actual_value_by_idx = {
            _order_key(tuple(int(idx) for idx in idxs)): float(value)
            for idxs, value in zip(
                actual_ic[ic_type], actual_ic_q[ic_type], strict=True
            )
        }
        expected_value_by_idx = {
            _order_key(tuple(int(idx) for idx in idxs)): float(value)
            for idxs, value in zip(*expected_ic[ic_type], strict=True)
        }

        if verbose:
            print("MISSING  ", {*expected_value_by_idx} - {*actual_value_by_idx})
            print("EXTRA    ", {*actual_value_by_idx} - {*expected_value_by_idx})

            print("", flush=True)

        assert {*expected_value_by_idx} == {*actual_value_by_idx}

        for key in {*expected_value_by_idx}.union({*actual_value_by_idx}):
            expected_value = expected_value_by_idx.get(key, numpy.nan)
            actual_value = actual_value_by_idx.get(key, numpy.nan)

            if verbose:
                print(key, f"{expected_value:.5f}", f"{actual_value:.5f}")

            assert numpy.isclose(expected_value, actual_value)


def test_compute_linear_displacement():
    coords = torch.tensor([[-1.0, 0.1, 0.0], [+0.0, 0.0, 0.0], [+1.0, 0.1, 0.0]])
    idxs = torch.tensor([[0, 1, 2, 0], [0, 1, 2, 1]])

    actual_value = _compute_linear_displacement(coords, idxs)

    # Displacement along axis 0 (in this case will be the +z axis) should be 0
    # as the angle between A->C and the +z axis is 90 degrees (i.e. dot product=0.0)
    #
    # Displacement along axis 1 (in this case will be the -y axis) will equal the
    # -y . a->c + -y . a->c
    a = 0.1 / float(math.sqrt(1.0 * 1.0 + 0.1 * 0.1))
    h = 1.0

    expected_value = torch.tensor([0.0, -2.0 * a / h])  # (cos θ = a / h)

    assert expected_value.shape == actual_value.shape
    assert torch.allclose(expected_value, actual_value)


@pytest.mark.parametrize(
    "smiles",
    ["C", "CC", "C#C", "CC#C", "CC#CC", "CCC#CCC", "CC#CC#CC", "CC(=O)c1ccc(cc1)C#N"],
)
def test_ric_from_coords(smiles):
    molecule = openff.toolkit.Molecule.from_smiles(smiles)
    molecule.generate_conformers(n_conformers=1)

    coords = torch.from_numpy(molecule.conformers[0].m_as("bohr"))

    _validate_ic(molecule, coords, verbose=True)
