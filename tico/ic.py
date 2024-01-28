"""Compute internal coordinate representations of molecules.

Notes:
    * This module is heavily inspired off of the ``internal`` module of ``geomeTRIC``.
      See the LICENSE-3RD-PARTY for license information.
"""
import abc
import dataclasses
import enum
import itertools
import logging
import typing

import networkx
import smee.geometry
import smee.utils
import torch

import tico.utils

_LOGGER = logging.getLogger(__name__)

# The linear threshold is taken from `geometric ==0.9.7.2`
_LINEAR_THRESHOLD = 0.95


class ICType(enum.Enum):
    """The supported primitive internal coordinate types."""

    DISTANCE = "DISTANCE"

    ANGLE = "ANGLE"
    LINEAR = "LINEAR"
    OUT_OF_PLANE = "OUT_OF_PLANE"

    DIHEDRAL = "DIHEDRAL"


ICDict = dict[ICType, torch.Tensor]
"""A dictionary of tensors partitioned by internal coordinate type."""

IC_ORDER = (
    ICType.DISTANCE,
    ICType.ANGLE,
    ICType.LINEAR,
    ICType.OUT_OF_PLANE,
    ICType.DIHEDRAL,
)
"""The order that internal coordinate types appear in a flattened representation."""

ConstraintDict = dict[ICType, tuple[torch.Tensor, torch.Tensor]]
"""A dictionary of constraints for each type of internal coordinate. It is of the form
``{ic_type: (idxs, values)}`` where ``idxs`` is a tensor of the indices of the atoms
involved in the internal coordinates to constrain and ``values`` is a tensor of the
values of that the internal coordinate should be constrained at."""


def _compute_linear_displacement(
    coords: torch.Tensor, idxs: torch.Tensor
) -> torch.Tensor:
    """Computes the displacement of the BA and BC unit vectors in the linear angle
    "ABC". The displacements are measured along two axes that are perpendicular to the
    AC unit vector.

    Args:
        coords: The cartesian coordinates of a coords with shape=(n_atoms, 3)
        idxs: A tensor containing the indices of the atoms in each linear angle
            (first three columns) and the index of the axis to compute the displacement
            along (last column) with shape=(n_linear_angles, 4).
    Returns:
        A tensor of the linear displacements.
    """

    vector_ab = coords[idxs[:, 0]] - coords[idxs[:, 1]]
    vector_ab = vector_ab / torch.norm(vector_ab, dim=1).unsqueeze(1)

    vector_cb = coords[idxs[:, 2]] - coords[idxs[:, 1]]
    vector_cb = vector_cb / torch.norm(vector_cb, dim=1).unsqueeze(1)

    vector_ca = coords[idxs[:, 2]] - coords[idxs[:, 0]]
    vector_ca = vector_ca / torch.norm(vector_ca, dim=1).unsqueeze(1)

    # Take the dot product of each row of ``vector_ca`` with the x, y and z axis
    # and find the index (0 = x, 1 = y, 2 = z) of the axis that is most perpendicular
    # to each row. This ensures we don't try and take the cross-product of two
    # co-linear vectors.
    #
    # This is the same approach taken by geomeTRIC, but a bit more vectorized.
    basis = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=coords.dtype
    )
    basis_index = torch.argmin((vector_ca @ basis).square(), dim=-1)

    axis_0 = basis[basis_index]

    axis_1 = torch.cross(vector_ca, axis_0)
    axis_1 = axis_1 / torch.norm(axis_1, dim=1).unsqueeze(1)

    axis_2 = torch.cross(vector_ca, axis_1)
    axis_2 = axis_2 / torch.norm(axis_2, dim=1).unsqueeze(1)

    return torch.where(
        idxs[:, 3] == 0,
        (vector_ab * axis_1).sum(dim=-1) + (vector_cb * axis_1).sum(dim=-1),
        (vector_ab * axis_2).sum(dim=-1) + (vector_cb * axis_2).sum(dim=-1),
    )


def _normal_vector(coordinates: torch.Tensor, angle_idxs: torch.Tensor) -> torch.Tensor:
    vector1 = coordinates[angle_idxs[:, 0]] - coordinates[angle_idxs[:, 1]]
    vector2 = coordinates[angle_idxs[:, 2]] - coordinates[angle_idxs[:, 1]]

    cross_product = torch.cross(vector1, vector2)

    return cross_product / torch.norm(cross_product, dim=-1).unsqueeze(-1)


def _detect_angles(
    coords: torch.Tensor, graph: networkx.Graph
) -> tuple[torch.Tensor, torch.Tensor]:
    """Detects any non-linear and linear angle degrees within a graph representation
    of a molecule for a given conformer.

    Returns:
        The indices of atoms that form non-linear angles with shape=(n_angles, 3)
        and the indices of atoms that form any linear angles with shape=(n_terms, 4).
    """

    angle_idxs = torch.tensor(
        [
            (a, b, c)
            for b in graph.nodes()
            for a in graph.neighbors(b)
            for c in graph.neighbors(b)
            if a < c
        ]
    )

    angles = smee.geometry.compute_angles(coords, angle_idxs)

    is_angle_linear = torch.abs(angles) >= torch.acos(torch.tensor(-_LINEAR_THRESHOLD))

    linear_angle_idxs = angle_idxs[is_angle_linear]
    linear_angle_idxs = torch.vstack(
        [
            torch.hstack(
                [
                    linear_angle_idxs,
                    torch.zeros(len(linear_angle_idxs), 1, dtype=torch.int64),
                ]
            ),
            torch.hstack(
                [
                    linear_angle_idxs,
                    torch.ones(len(linear_angle_idxs), 1, dtype=torch.int64),
                ]
            ),
        ]
    )

    angle_idxs = angle_idxs[~is_angle_linear]

    return angle_idxs, linear_angle_idxs


def _detect_out_of_plane_angles(
    coords: torch.Tensor,
    graph: networkx.Graph,
    angle_idxs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Detects any out of plane angles within a graph representation of a molecule for
    a given conformer.

    Args:
        coords: The coordinates of the molecule with shape=(n_atoms, 3)
        graph: The associated molecule stored in a ``networkx`` graph object.
        angle_idxs: A tensor storing the indices of atoms that form any non-linear
            angles with shape=(n_angles, 3)

    Returns:
        The indices of atoms that form any non-linear angles with shape=(n_angles, 3)
        and a tensor storing the indices of atoms that form any out of plane (i.e.
        improper) angles with shape=(n_terms, 4).
    """

    improper_tuples = [
        (a, b, c, d)
        for b in graph.nodes()
        for a in graph.neighbors(b)
        for c in graph.neighbors(b)
        for d in graph.neighbors(b)
        if a < c < d
    ]

    out_of_plane_tuples, angles_to_remove = [], set()

    for a, b, c, d in improper_tuples:
        improper_idxs = torch.tensor(
            [(b, i, j, k) for i, j, k in sorted(itertools.permutations([a, c, d], 3))]
        )

        angles_a = smee.geometry.compute_angles(coords, improper_idxs[:, (0, 1, 2)])
        angles_b = smee.geometry.compute_angles(coords, improper_idxs[:, (1, 2, 3)])

        is_out_of_plane = (
            (torch.abs(torch.cos(angles_a)) <= _LINEAR_THRESHOLD)
            & (torch.abs(torch.cos(angles_b)) <= _LINEAR_THRESHOLD)
            & (
                torch.abs(
                    torch.sum(
                        _normal_vector(coords, improper_idxs[:, (0, 1, 2)])
                        * _normal_vector(coords, improper_idxs[:, (1, 2, 3)]),
                        dim=-1,
                    )
                )
                > _LINEAR_THRESHOLD
            )
        )

        if not torch.any(is_out_of_plane):
            continue

        out_of_plane_tuples.append(improper_idxs[0, :])

        angle_to_remove = tuple(int(i) for i in improper_idxs[0, (1, 0, 2)])

        angles_to_remove.add(angle_to_remove)
        angles_to_remove.add(tuple(reversed(angle_to_remove)))

    if len(out_of_plane_tuples) == 0:
        return angle_idxs, torch.tensor([], dtype=torch.int64)

    angle_mask = torch.tensor(
        [tuple(int(i) for i in row) not in angles_to_remove for row in angle_idxs]
    )

    out_of_plane_idxs = torch.vstack(out_of_plane_tuples)
    angle_idxs = angle_idxs[angle_mask]

    return angle_idxs, out_of_plane_idxs


def _detect_dihedrals(
    coords: torch.Tensor,
    graph: networkx.Graph,
    linear_angle_idxs: torch.Tensor,
) -> torch.Tensor:
    """Detects any dihedral degrees within a graph representation of a molecule for
    a given conformer.

    Returns:
        The indices of atoms that form any dihedrals with shape=(n_dihedrals, 3).
    """

    # Compute all 'standard' dihedrals excluding linear dihedrals.
    dihedral_idxs = torch.tensor(
        [
            (a, b, c, d)
            for (b, c) in graph.edges()
            for a in graph.neighbors(b)
            for d in graph.neighbors(c)
            if a != c and d != b
        ]
    )

    if len(dihedral_idxs) == 0:
        return torch.tensor([], dtype=torch.int64)

    # Also compute dihedrals where the central bond is actually a linear chain
    # of atoms rather than a single bond.
    graph = graph.copy()

    linear_chain_edges = set()

    for node in {int(i) for i in linear_angle_idxs[:, 1]}:
        chain_edge = tuple(graph.neighbors(node))

        graph.add_edge(*chain_edge)
        graph.remove_node(node)

        linear_chain_edges.add(chain_edge)

    chain_dihedral_idxs = torch.tensor(
        [
            (a, b, c, d)
            for (b, c) in graph.edges()
            for a in graph.neighbors(b)
            for d in graph.neighbors(c)
            if a != c
            and d != b
            and (a, b) not in linear_chain_edges
            and (b, a) not in linear_chain_edges
            and (c, d) not in linear_chain_edges
            and (d, c) not in linear_chain_edges
        ]
    )

    if len(chain_dihedral_idxs) > 0:
        dihedral_idxs = torch.unique(
            torch.vstack([dihedral_idxs, chain_dihedral_idxs]), dim=0
        )

    angles_a = smee.geometry.compute_angles(coords, dihedral_idxs[:, (0, 1, 2)])
    angles_b = smee.geometry.compute_angles(coords, dihedral_idxs[:, (1, 2, 3)])

    # Remove linear dihedrals
    dihedral_mask = (torch.abs(torch.cos(angles_a)) < _LINEAR_THRESHOLD) & (
        torch.abs(torch.cos(angles_b)) < _LINEAR_THRESHOLD
    )

    dihedral_idxs = dihedral_idxs[dihedral_mask]

    return dihedral_idxs


def _flatten_ic(ic: ICDict) -> torch.Tensor:
    """Flatten a dictionary of internal coordinates into a single tensor."""
    return torch.cat([ic[ic_type] for ic_type in IC_ORDER if ic_type in ic])


def _match_constr(constr: ConstraintDict, ic_idxs: ICDict) -> torch.Tensor:
    """Find the indices of the internal coordinates that correspond to constrained
    DoF."""
    constr_to_ic_dx = []
    count = 0

    for ic_type in IC_ORDER:
        if ic_type in constr and ic_type not in ic_idxs:
            raise ValueError(f"constraint type {ic_type} not found in RIC.")
        if ic_type not in ic_idxs:
            continue
        if ic_type not in constr:
            count += len(ic_idxs[ic_type])
            continue

        match = torch.all(constr[ic_type][0][:, None] == ic_idxs[ic_type], dim=2)

        match_idxs = torch.where(
            torch.any(match, dim=1), torch.argmax(match.int(), dim=1), -1
        )
        assert torch.all(match_idxs > -1)

        constr_to_ic_dx.append(match_idxs + count)
        count += len(ic_idxs[ic_type])

    return torch.cat(constr_to_ic_dx)


def _compute_q(coords_x: torch.Tensor, ic_idxs: ICDict) -> ICDict:
    """Maps a set of cartesian coordinates to a set of internal coordinates.

    Args:
        coords_x: The coordinates with ``shape=(n_atoms, 3)``.
        ic_idxs: The indices of the atoms involved in each type of internal coordinate.

    Returns:
        A dictionary of the form ``coords_q[ic_type] = value`` where ``value`` is
        a tensor of the internal coordinate values.
    """

    ic_value_fn = {
        ICType.DISTANCE: lambda *args: smee.geometry.compute_bond_vectors(*args)[1],
        ICType.ANGLE: smee.geometry.compute_angles,
        ICType.LINEAR: _compute_linear_displacement,
        ICType.OUT_OF_PLANE: smee.geometry.compute_dihedrals,
        ICType.DIHEDRAL: smee.geometry.compute_dihedrals,
    }
    return {
        ic_type: ic_value_fn[ic_type](coords_x, ic_atom_idxs)
        for ic_type, ic_atom_idxs in ic_idxs.items()
    }


def _compute_dq(q_a: ICDict, q_b: ICDict) -> torch.Tensor:
    """Compute the difference between two sets of coordinates in terms of internal
    coordinates.

    Args:
        q_a: The first set of internal coordinates.
        q_b: The second set of internal coordinates.

    Returns:
        The internal coordinate displacements with ``shape=(n_ic,)``.
    """
    assert {*q_a} == {*q_b}

    deltas = []

    for ic_type in IC_ORDER:
        if ic_type not in q_a:
            continue

        assert len(q_b[ic_type]) == len(q_a[ic_type])
        delta = q_b[ic_type] - q_a[ic_type]

        if ic_type in {ICType.OUT_OF_PLANE, ICType.DIHEDRAL}:
            delta_plus = delta + 2.0 * torch.pi
            delta_minus = delta - 2.0 * torch.pi

            delta = torch.where(
                torch.abs(delta) > torch.abs(delta_plus), delta_plus, delta
            )
            delta = torch.where(
                torch.abs(delta) > torch.abs(delta_minus), delta_minus, delta
            )

        deltas.append(delta)

    return torch.cat(deltas)


def _dq_to_x(
    coords_x: torch.Tensor, dq: torch.Tensor, ic: typing.Union["RIC", "DLC"]
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Perturb a set of cartesian coordinates by a set of internal coordinate
    displacements.

    Args:
        coords_x: The coordinates with ``shape=(n_atoms, 3)``.
        dq: The internal coordinate displacements with ``shape=(n_ic,)``.

    Returns:
        The perturbed cartesian (with ``shape=(n_atoms, 3)``) and internal
        coordinates (with ``shape=(n_ic,)``), and a boolean indicating whether the
        perturbation was successful.
    """

    iteration, failures = 0, 0

    damp = 1.0

    coords_iter_1, coords_final = None, None

    dq_delta_prev = None

    while True:
        iteration += 1

        b_matrix = ic.compute_b(coords_x)
        g_matrix_inv = tico.utils.pinv(b_matrix @ b_matrix.T)

        dx = damp * torch.linalg.multi_dot([b_matrix.T, g_matrix_inv, dq])

        coords_x_new = coords_x + dx.reshape(-1, 3)
        coords_q_new = ic.compute_q(coords_x_new)

        if iteration == 1:
            coords_final = coords_iter_1 = coords_x_new, coords_q_new

        dq_actual = ic.compute_dq(coords_x, coords_x_new)
        dq_delta = torch.linalg.norm(dq - dq_actual)

        rmsd = torch.sqrt(torch.mean((coords_x_new - coords_x) ** 2))

        if dq_delta_prev is not None:
            if dq_delta > dq_delta_prev:
                damp *= 0.5
                failures += 1
            else:
                damp = min(damp * 1.2, 1.0)
                failures = 0

                coords_final = coords_x_new, coords_q_new
                dq_delta_prev = dq_delta
        else:
            dq_delta_prev = dq_delta

        if (rmsd < 1e-6 or dq_delta < 1e-6) or failures >= 5 or iteration >= 50:
            converged = dq_delta <= 1e-1

            coords_x, coords_q = coords_final if converged else coords_iter_1
            return coords_x, coords_q, converged

        dq = dq - dq_actual
        coords_x, coords_q = coords_x_new, coords_q_new


def _dq_to_x_cached(
    coords_x: torch.Tensor,
    dq: torch.Tensor,
    ic: typing.Union["RIC", "DLC"],
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """A cached version of the ``_dq_to_x`` function.

    While a bit of a hack, this can significantly speed up ops like the brent
    line search.
    """
    if ic._dq_cache is not None:
        (
            cached_coords_x,
            cached_dq,
            cached_coords_x_new,
            cached_coords_q_new,
            cached_q_converged,
        ) = ic._dq_cache

        if (
            torch.linalg.norm(cached_coords_x - coords_x) < 1e-10
            and torch.linalg.norm(cached_dq - dq) < 1e-10
        ):
            return cached_coords_x_new, cached_coords_q_new, cached_q_converged

    coords_x_new, coords_q_new, q_converged = _dq_to_x(coords_x, dq, ic)
    ic._dq_cache = (coords_x, dq, coords_x_new, coords_q_new, q_converged)

    return coords_x_new, coords_q_new, q_converged


@dataclasses.dataclass
class IC(abc.ABC):
    """A base class for internal coordinate representations."""

    _dq_cache = None
    """A cache of the last internal coordinate optimization."""

    idxs: ICDict
    """The indices of the atoms involved in each type of internal coordinate."""

    @abc.abstractmethod
    def compute_b(self, coords_x: torch.Tensor) -> torch.Tensor:
        """Computes jacobian of the internal coordinates with respect to the cartesian
        coordinates.

        This is the B matrix in the Pulay-Fogarasi-Pulay approach.

        Args:
            coords_x: The coordinates with ``shape=(n_atoms, 3)``.

        Returns:
            The jacobian, with ``shape=(n_ic, n_atoms * 3)``.
        """

    @abc.abstractmethod
    def compute_q(self, coords_x: torch.Tensor) -> torch.Tensor:
        """Maps a set of cartesian coordinates to a set of internal coordinates.

        Args:
            coords_x: The coordinates with ``shape=(n_atoms, 3)``.

        Returns:
            The flattened internal coordinate values with ``shape=(n_ic,)``.
        """

    @abc.abstractmethod
    def compute_dq(
        self, coords_x_a: torch.Tensor, coords_x_b: torch.Tensor
    ) -> torch.Tensor:
        """Compute the difference between two sets of coordinates in terms of internal
        coordinates.

        Args:
            coords_x_a: The first coordinates with ``shape=(n_atoms, 3)``.
            coords_x_b: The second coordinates with ``shape=(n_atoms, 3)``.

        Returns:
            The internal coordinate displacements with ``shape=(n_ic,)``.
        """

    @abc.abstractmethod
    def guess_hess_q(
        self, coords_x: torch.Tensor, atomic_nums: torch.Tensor
    ) -> torch.Tensor:
        """Build an approximate Hessian that roughly follows Schlegel's guidelines.

        Args:
            coords_x: The coordinates with ``shape=(n_atoms, 3)``.
            atomic_nums: The atomic number of each atom with ``shape=(n_atoms,)``.

        Returns:
            The approximate Hessian with ``shape=(n_ic, n_ic)``.
        """

    @abc.abstractmethod
    def dq_to_x(
        self, coords_x: torch.Tensor, dq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Perturb a set of cartesian coordinates by a set of internal coordinate
        displacements.

        Args:
            coords_x: The coordinates with ``shape=(n_atoms, 3)``.
            dq: The internal coordinate displacements with ``shape=(n_ic,)``.

        Returns:
            The perturbed cartesian (with ``shape=(n_atoms, 3)``) and internal
            coordinates (with ``shape=(n_ic,)``), and a boolean indicating whether the
            perturbation was successful.
        """


@dataclasses.dataclass
class RIC(IC):
    """A redundant internal coordinate representation."""

    @classmethod
    def from_coords(cls, coords_x: torch.Tensor, bond_idxs: torch.Tensor) -> "RIC":
        """Projects a set of cartesian coordinates onto a reduced set of delocalized
        internal coordinates.

        Args:
            coords_x: The coordinates with ``shape=(n_atoms, 3)``.
            bond_idxs: The atoms involved in each bond with ``shape=(n_bonds, 2)``.

        Returns:
            The internal coordinate representation.
        """
        coords_x = coords_x.double()

        graph = networkx.Graph(bond_idxs.detach().tolist())

        angle_idxs, linear_angle_idxs = _detect_angles(coords_x, graph)
        angle_idxs, out_of_plane_idxs = _detect_out_of_plane_angles(
            coords_x, graph, angle_idxs
        )

        dihedral_idxs = _detect_dihedrals(coords_x, graph, linear_angle_idxs)

        return_value = {
            ICType.DISTANCE: bond_idxs,
            ICType.ANGLE: angle_idxs,
            ICType.LINEAR: linear_angle_idxs,
            ICType.OUT_OF_PLANE: out_of_plane_idxs,
            ICType.DIHEDRAL: dihedral_idxs,
        }
        return RIC(
            {key: value for key, value in return_value.items() if len(value) > 0}
        )

    def compute_b(self, coords_x: torch.Tensor) -> torch.Tensor:
        b_matrix = torch.func.jacfwd(self.compute_q)(coords_x)
        return b_matrix.reshape(len(b_matrix), -1)

    def compute_q(self, coords_x: torch.Tensor) -> torch.Tensor:
        return _flatten_ic(_compute_q(coords_x, self.idxs))

    def compute_dq(
        self, coords_x_a: torch.Tensor, coords_x_b: torch.Tensor
    ) -> torch.Tensor:
        return _compute_dq(
            _compute_q(coords_x_a, self.idxs), _compute_q(coords_x_b, self.idxs)
        )

    def guess_hess_q(
        self, coords_x: torch.Tensor, atomic_nums: torch.Tensor
    ) -> torch.Tensor:
        import tico.hess

        return tico.hess.guess_hess_q(coords_x, self.idxs, atomic_nums)

    def dq_to_x(
        self, coords_x: torch.Tensor, dq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        return _dq_to_x_cached(coords_x, dq, self)


@dataclasses.dataclass
class DLC(IC):
    """A delocalized internal coordinates representation"""

    v: torch.Tensor
    """The projection of RIC onto the non-redundant basis."""
    constr: ConstraintDict | None = None
    """Constraints on the internal coordinates."""

    @classmethod
    def _project_constr(
        cls, v: torch.Tensor, constr: ConstraintDict, ric: RIC
    ) -> torch.Tensor:
        """Attempts to project the constrained primitive internal coordinates onto the
        non-redundant basis.

        Notes:
            This uses the 'old' method used in geomeTRIC, which is a bit more
            battle tested than the more 'correct new' method.
        """

        n_v = v.shape[1]

        constr_to_ic_idx = _match_constr(constr, ric.idxs)

        v_constr = v[constr_to_ic_idx, :]
        v_constr = v_constr / torch.linalg.norm(v_constr, dim=-1, keepdim=True)

        v_proj = v @ v_constr.T
        v_proj = v_proj / torch.linalg.norm(v_proj, dim=0, keepdim=True)

        v = torch.hstack((v_proj, v))

        threshold = 1e-6

        while True:
            u = []
            for i in range(v.shape[1]):
                v_i = v[:, i].flatten()
                u.append(v_i.clone())

                for u_i in u[:-1]:
                    u[-1] = u[-1] - u_i * torch.dot(u_i, v_i)

                if torch.linalg.norm(u[-1]) < threshold:
                    u = u[:-1]
                    continue

                u[-1] = u[-1] / torch.linalg.norm(u[-1])

            if len(u) > n_v:
                threshold *= 10.0
            elif len(u) == n_v:
                break
            elif len(u) < n_v:
                raise RuntimeError("Gram-Schmidt orthogonalization failed")

        return torch.vstack(u).T

    @classmethod
    def from_coords(
        cls,
        coords_x: torch.Tensor,
        bond_idxs: torch.Tensor,
        constr: ConstraintDict | None = None,
    ) -> "DLC":
        """Projects a set of cartesian coordinates onto a reduced set of delocalized
        internal coordinates.

        Args:
            coords_x: The coordinates with shape=(n_atoms, 3).
            bond_idxs: The atoms involved in each bond with shape=(n_bonds, 2).
            constr: A dictionary of constraints on the internal coordinates. This
                dictionary should be of the form ``constr[ic_type] = (idxs, values)``
                where ``idxs`` is a tensor of the indices of the atoms involved in the
                internal coordinate and ``values`` is a tensor of the values to
                constrain the internal coordinate at.

        Returns:
            The internal coordinate representation.
        """
        coords_x = coords_x.double()

        ric = RIC.from_coords(coords_x, bond_idxs)

        b_matrix = ric.compute_b(coords_x)
        g_matrix = b_matrix @ b_matrix.T

        eig_vals, eig_vecs = torch.linalg.eigh(g_matrix)
        eig_mask = torch.abs(eig_vals) > 1e-6

        v = eig_vecs[:, eig_mask]

        if constr is not None and len(constr) == 0:
            constr = None
        if constr is not None:
            v = cls._project_constr(v, constr, ric)

        return cls(ric.idxs, v, constr)

    def compute_b(self, coords_x: torch.Tensor) -> torch.Tensor:
        def compute_q(c: torch.Tensor) -> torch.Tensor:
            return _flatten_ic(_compute_q(c, self.idxs))

        b_matrix = torch.func.jacfwd(compute_q)(coords_x)
        b_matrix_proj = torch.tensordot(self.v, b_matrix, dims=([0], [0]))
        return b_matrix_proj.reshape(len(b_matrix_proj), -1)

    def compute_q(self, coords_x: torch.Tensor) -> torch.Tensor:
        return _flatten_ic(_compute_q(coords_x, self.idxs)) @ self.v

    def compute_dq(
        self, coords_x_a: torch.Tensor, coords_x_b: torch.Tensor
    ) -> torch.Tensor:
        return (
            _compute_dq(
                _compute_q(coords_x_a, self.idxs), _compute_q(coords_x_b, self.idxs)
            )
            @ self.v
        )

    def compute_constr_delta(self, coords_x: torch.Tensor) -> torch.Tensor:
        """Compute the difference between the current internal coordinate values and the
        constrained values."""
        constr_idxs = {ic_type: idxs for ic_type, (idxs, _) in self.constr.items()}
        constr_vals = {ic_type: vals for ic_type, (_, vals) in self.constr.items()}

        coords_q = _compute_q(coords_x, constr_idxs)

        return _compute_dq(constr_vals, coords_q)

    def guess_hess_q(
        self, coords_x: torch.Tensor, atomic_nums: torch.Tensor
    ) -> torch.Tensor:
        import tico.hess

        hess_q = tico.hess.guess_hess_q(coords_x, self.idxs, atomic_nums)
        return torch.linalg.multi_dot([self.v.T, hess_q, self.v])

    def augment_hess_q(
        self, coords_x: torch.Tensor, grad_q: torch.Tensor, hess_q: torch.Tensor
    ):
        """Augment a hessian with extra dimensions corresponding to the constrained
        DoF.

        Args:
            coords_x: The coordinates with ``shape=(n_atoms, 3)``.
            grad_q: The gradient of the internal coordinates with ``shape=(n_ic,)``.
            hess_q: The hessian of the internal coordinates with ``shape=(n_ic, n_ic)``.
        """

        constr_to_ic_idx = _match_constr(self.constr, self.idxs)

        ni = len(grad_q)
        nc = len(constr_to_ic_idx)

        diag_idxs = list(range(len(constr_to_ic_idx)))

        ct = smee.utils.zeros_like((nc, ni), coords_x)
        ct[diag_idxs, diag_idxs] = 1.0 / self.v[constr_to_ic_idx, diag_idxs]

        nt = ni + nc

        constr_diff = -self.compute_constr_delta(coords_x)
        # au/rad (about 0.16 A / 17 deg)
        # constr_diff[:nc] = torch.clamp(constr_diff[:nc], -0.3, 0.3)

        hess_q_constr = smee.utils.zeros_like((nt, nt), coords_x)
        hess_q_constr[0:ni, 0:ni] = hess_q[:, :]
        hess_q_constr[ni:nt, 0:ni] = ct[:, :]
        hess_q_constr[0:ni, ni:nt] = ct.T[:, :]

        grad_q_constr = smee.utils.zeros_like(nt, coords_x)
        grad_q_constr[0:ni] = grad_q[:]
        grad_q_constr[ni:nt] = -constr_diff[:]

        return grad_q_constr, hess_q_constr

    def dq_to_x(
        self,
        coords_x: torch.Tensor,
        dq: torch.Tensor,
        constr_tol: float = 0.1,
        enforce_constr: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        coords_x, coords_q, q_converged = _dq_to_x_cached(coords_x, dq, self)

        if self.constr is None:
            return coords_x, coords_q, q_converged

        constr_delta = self.compute_constr_delta(coords_x)

        if enforce_constr and torch.linalg.norm(constr_delta) >= constr_tol:
            return self.enforce_constraints(coords_x, constr_tol)

        return coords_x, coords_q, q_converged

    def project_grad_x(
        self, coords_x: torch.Tensor, grad_x: torch.Tensor
    ) -> torch.Tensor:
        """Project out the components of the internal coordinate gradient along the
        constrained degrees of freedom."""

        if self.constr is None:
            return grad_x

        b_matrix = self.compute_b(coords_x)
        g_matrix_inv = tico.utils.pinv(b_matrix @ b_matrix.T)

        n_constr = sum(len(v) for i, v in self.constr.values())

        grad_q = torch.linalg.multi_dot([g_matrix_inv, b_matrix, grad_x])
        grad_q[:n_constr] = 0.0

        return torch.linalg.multi_dot([b_matrix.T, grad_q])

    def enforce_constraints(
        self, coords_x: torch.Tensor, tol: float = 1e-6
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Modify cartesian coordinates to enforce any internal coordinate constraints.

        Args:
            coords_x: The coordinates with shape=(n_atoms, 3).
            tol: The tolerance for the constraint satisfaction.

        Returns:
            The modified coordinates and internal coordinates, and a boolean indicating
            whether the conversion of cartesian to internal coordinates was successful.
        """
        iteration = 0

        constr_to_ic_idx = _match_constr(self.constr, self.idxs)

        dq_norm = 0.0
        dq_norm_best, soln_best = None, None

        while True:
            dq = smee.utils.zeros_like(self.v.shape[-1], self.v)

            # because of how we construct v, the first n_v columns should be the
            # projected constrained DoF
            constr_delta = -self.compute_constr_delta(coords_x)

            dq[: len(constr_delta)] = constr_delta
            dq[: len(constr_delta)] /= self.v[
                constr_to_ic_idx, list(range(len(constr_delta)))
            ]

            dq_norm_prev = dq_norm
            dq_norm = torch.linalg.norm(dq)

            soln = self.dq_to_x(coords_x, dq, enforce_constr=False)
            coords_x = soln[0]

            if dq_norm_best is None or dq_norm < dq_norm_best:
                dq_norm_best, soln_best = dq_norm, soln

            if dq_norm < tol:
                return soln

            if iteration > 0 and dq_norm > dq_norm_prev:
                _LOGGER.warning("constraint satisfaction failed to converge")
                return soln_best

            iteration += 1
