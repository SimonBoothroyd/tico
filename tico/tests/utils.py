"""Test utilities"""

import openff.toolkit
import openmm
import torch


def energy_fn(context: openmm.Context):
    """Create an energy function from an OpenMM context."""

    def compute(coords):
        coords = coords.numpy()
        context.setPositions(coords.reshape(-1, 3) * openmm.unit.bohr)

        state = context.getState(getEnergy=True, getForces=True)

        energy = state.getPotentialEnergy() / openmm.unit.AVOGADRO_CONSTANT_NA
        gradient = -state.getForces(asNumpy=True) / openmm.unit.AVOGADRO_CONSTANT_NA

        energy = energy.value_in_unit(openmm.unit.hartree)
        gradient = gradient.value_in_unit(
            openmm.unit.hartree / openmm.unit.bohr
        ).flatten()

        return torch.tensor(energy), torch.tensor(gradient)

    return compute


def create_test_case(smiles: str):
    """Prepare a molecule for optimization from its SMILES description."""
    ff = openff.toolkit.ForceField("openff_unconstrained-2.1.0.offxml")

    mol = openff.toolkit.Molecule.from_smiles(smiles)
    mol.generate_conformers(n_conformers=1)

    bond_idxs = torch.tensor(
        [[bond.atom1_index, bond.atom2_index] for bond in mol.bonds]
    )
    atomic_nums = torch.tensor([atom.atomic_number for atom in mol.atoms])

    coords = torch.tensor(mol.conformers[0].m_as("bohr").tolist())

    system = ff.create_openmm_system(mol.to_topology())

    context = openmm.Context(
        system,
        openmm.VerletIntegrator(1.0),
        openmm.Platform.getPlatformByName("Reference"),
    )

    return coords, atomic_nums, bond_idxs, energy_fn(context)
