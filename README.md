<h1 align="center">tico</h1>

<p align="center">Torch-based internal coordinate geometry optimization.</p>

<p align="center">
  <a href="https://github.com/SimonBoothroyd/tico/actions?query=workflow%3Aci">
    <img alt="ci" src="https://github.com/SimonBoothroyd/tico/actions/workflows/ci.yaml/badge.svg" />
  </a>
  <a href="https://codecov.io/gh/SimonBoothroyd/tico/branch/main">
    <img alt="coverage" src="https://codecov.io/gh/SimonBoothroyd/tico/branch/main/graph/badge.svg" />
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="license" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
</p>

---

The `tico` framework provides utilities for optimizing the geometry of molecules in
internal coordinates. It is heavily based off of [geomeTRIC](https://github.com/leeping/geomeTRIC), but aims to improve
the performance using PyTorch, and bypassing some of the QA checks. For a more robust
option, consider using [geomeTRIC](https://github.com/leeping/geomeTRIC) instead.

Because `tico` is heavily based off of `geomeTRIC`, please consider citing the original
package if you use `tico` in your work:

```text
@article{wang2016geometry,
  title={Geometry optimization made simple with translation and rotation coordinates},
  author={Wang, Lee-Ping and Song, Chenchen},
  journal={The Journal of chemical physics},
  volume={144},
  number={21},
  year={2016},
  publisher={AIP Publishing}
}
```

## Installation

This package can be installed using `conda` (or `mamba`, a faster version of `conda`):

```shell
mamba install -c conda-forge tico
```

## Getting Started

To geometry of a molecule can be optimized using the `tico.opt.optimize` function. It
takes as input and initial set of cartesian coordinates, and a function that returns
the energy and gradient of the molecule at a given set of coordinates.

Creating a molecule to optimize can be easily done using the `openff.toolkit` package:

```python
import openff.toolkit
import torch

ff = openff.toolkit.ForceField("openff_unconstrained-2.1.0.offxml")

mol = openff.toolkit.Molecule.from_smiles("CCO")
mol.generate_conformers(n_conformers=1)

bond_idxs = torch.tensor(
    [[bond.atom1_index, bond.atom2_index] for bond in mol.bonds]
)
coords = torch.tensor(mol.conformers[0].m_as("bohr")).double()
```

An internal coordinate representation of the molecule can then be created using the
helpers in the `tico.ic` module:

```python
import tico.ic

# Create a primitive internal coordinates representation
ic = tico.ic.RIC.from_coords(coords, bond_idxs)
# Or a usually more efficient delocalized internal coordinates representation
ic = tico.ic.DLC.from_coords(coords, bond_idxs)

# If using the delocalized internal coordinates, optional constraints can be added.
# For example, to fix the distance between atoms 0 and 1 to 2.0 bohr:
constr = {tico.ic.ICType.DISTANCE: (torch.tensor([[0, 1]]), torch.tensor([2.0]))}
ic = tico.ic.DLC.from_coords(coords, bond_idxs, constr)
```

An example energy function that uses OpenMM may look like:

```python
import openmm
import openmm.unit
import torch

system = ff.create_openmm_system(mol.to_topology())

context = openmm.Context(
    system,
    openmm.VerletIntegrator(1.0),
    openmm.Platform.getPlatformByName("Reference"),
)

def energy_fn(c):
    c = c.numpy().reshape(-1, 3) * openmm.unit.bohr
    context.setPositions(c)

    state = context.getState(getEnergy=True, getForces=True)

    energy = state.getPotentialEnergy() / openmm.unit.AVOGADRO_CONSTANT_NA
    gradient = -state.getForces(asNumpy=True) / openmm.unit.AVOGADRO_CONSTANT_NA

    energy = energy.value_in_unit(openmm.unit.hartree)
    gradient = gradient.value_in_unit(openmm.unit.hartree / openmm.unit.bohr).flatten()

    return torch.tensor(energy), torch.tensor(gradient)
```

The optimization can then be performed using:

```python
import tico.opt

atomic_nums = torch.tensor([atom.atomic_number for atom in mol.atoms])

history, converged = tico.opt.optimize(coords, ic, energy_fn, atomic_nums)
assert converged

coords_final = history[-1].coords
```

The `tico.td` module contains additional helpers for performing torsion scans around
given bonds:

```python
import tico.td

# scan using torsiondrive with wavefront propagation
params = tico.td.Params(driver=tico.td.WFP())
# OR for faster (but likely higher energy structures), just do a linear scan
params = tico.td.Params(driver=tico.td.Simple())

results = tico.td.torsion_drive(
    coords, bond_idxs, dihedral, energy_fn, atomic_nums, params
)
```

The energies of the torsion scan can be easily plotted:

```python

angles = sorted(results)
energies = [results[angle][1] for angle in angles]

from matplotlib import pyplot

pyplot.plot(angles, energies)
pyplot.xlabel("Angle [deg]")
pyplot.ylabel("Energy [Eh]")
pyplot.legend()
pyplot.show()
```

and the optimized geometries can be extracted:

```python
import openff.units

mol._conformers = [
    coords.numpy() * openff.units.unit.bohr for coords, _ in results.values()
]
mol.to_file("td.xyz", "XYZ")
```

## License

The main package is release under the [MIT license](LICENSE). Parts of the package are
largely inspired by [geomeTRIC](https://github.com/leeping/geomeTRIC), see the [LICENSE-3RD-PARTY](LICENSE-3RD-PARTY) file for the
license of the original code.
