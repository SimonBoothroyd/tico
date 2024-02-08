import pytest
import smee.geometry
import torch

import tico.ic
import tico.opt
import tico.tests.utils


@pytest.mark.parametrize("coord_sys", [tico.ic.RIC, tico.ic.DLC])
def test_opt_ric(coord_sys):
    coords_x, atomic_nums, bond_idxs, energy_fn = tico.tests.utils.create_test_case(
        "CC"
    )

    ic = coord_sys.from_coords(coords_x, bond_idxs)

    history, converged = tico.opt.optimize(coords_x, ic, energy_fn, atomic_nums)
    assert converged

    expected_energy = 0.0016070131551260988
    expected_grad_x = torch.tensor(
        [
            [-1.3046894717683241e-05, -4.929390811309805e-05, 9.04099523685593e-05],
            [8.66276149380544e-06, -1.3074926073320704e-05, -0.00012464827807203867],
            [-1.681245057597204e-05, -6.562533735831569e-06, -8.0771286793329e-06],
            [-1.8826956917417374e-05, 4.122129987314577e-05, 1.4342807965415285e-06],
            [-1.9048761812701595e-06, 7.736200242572962e-06, -2.655973816738187e-05],
            [1.7911334676965396e-05, -1.839842845092182e-05, 2.2951710655844116e-05],
            [2.9021509732307862e-05, 3.9007748822467705e-05, 5.313455250245805e-06],
            [-5.004427510735992e-06, -6.354525650142754e-07, 3.917574584756259e-05],
        ],
        dtype=torch.float64,
    ).flatten()

    energy, grad_x = energy_fn(history[-1].coords_x)

    torch.isclose(energy, torch.tensor(expected_energy), atol=1e-7)
    torch.allclose(grad_x, expected_grad_x, atol=1e-7)


def test_opt_constr():
    coords_x, atomic_nums, bond_idxs, energy_fn = tico.tests.utils.create_test_case(
        "CC"
    )

    bond = torch.tensor([[0, 1]])
    expected_dist = torch.tensor([1.6], dtype=torch.float64)

    constr = {tico.ic.ICType.DISTANCE: (bond, expected_dist)}
    ic = tico.ic.DLC.from_coords(coords_x, bond_idxs, constr)

    history, converged = tico.opt.optimize(coords_x, ic, energy_fn, atomic_nums)
    assert converged

    dist = smee.geometry.compute_bond_vectors(history[-1].coords_x, bond)[-1]
    assert torch.isclose(dist, expected_dist, atol=1e-7)
