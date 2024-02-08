import pytest
import torch

import tico.td
import tico.tests.utils


@pytest.mark.parametrize(
    "params",
    [
        tico.td.Params(grid_spacing=90, driver=tico.td.Simple()),
        tico.td.Params(grid_spacing=90, driver=tico.td.WFP()),
    ],
)
def test_torsion_drive(params):
    coords_x, atomic_nums, bond_idxs, energy_fn = tico.tests.utils.create_test_case(
        "CC"
    )

    result = tico.td.torsion_drive(
        coords_x, bond_idxs, (2, 0, 1, 5), energy_fn, atomic_nums, params
    )
    energies = torch.tensor([result[angle][1] for angle in sorted(result)])

    expected_energies = torch.tensor([0.004945, 0.009126, 0.004945, 0.001607])
    assert energies.shape == expected_energies.shape
    assert torch.allclose(energies, expected_energies, atol=1e-5)
