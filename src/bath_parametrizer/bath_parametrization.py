"""Parametrizing the bath automatically using character
table and generators of an abelian/non-abelian point group
(Pyqcm formalism).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from point_groups import all_point_groups


class BathParametrizer:
    """Given a cluster and an abelian point group, this class computes
    the symmetry-adapted coordinates of the cluster from which a discrete
    bath of non-correlated orbitals can be parametrized in the Pyqcm
    formalism.

    The conventions used follow the Bilbao Crystallographic Server.
    """

    def __init__(self, site_positions: np.ndarray, point_group: str) -> None:
        """Determines the point group from string and associates it to
        hardcoded point group data.
        """
        self.point_group = all_point_groups[point_group]
        self.positions = site_positions
        self.operations = self.point_group.operations
        self.character_table = self.point_group.character_table
        self.classes_multiplicity = self.point_group.classes_multiplicity
        self.order = len(self.operations)

    def _get_center_of_mass(self) -> np.ndarray:
        """Using center of mass of inputed cluster sites in order
        to apply the vectorial representation of the group elements.
        """
        return self.positions - np.average(self.positions, axis=0)

    def _get_irrep_multiplicity_from_repr(self) -> np.ndarray:
        """Applies the characters orthogonality relation to compute the
        multiplicity of the irreps. within a given representation.
        """
        repr_characters = np.zeros(len(self.operations), dtype=complex)
        for idx, op in enumerate(self.operations.values()):
            permutation_repr = self.build_permutation_repr(op)
            repr_characters[idx] = np.trace(permutation_repr)

        comps = np.zeros(len(self.character_table), dtype=complex)
        for idx, (_, irrep_chars) in enumerate(self.character_table.items()):
            extended_irrep = np.repeat(irrep_chars, self.classes_multiplicity)
            mj = 1 / self.order * np.sum(extended_irrep.conj() * repr_characters)
            comps[idx] = mj

        return comps

    def build_permutation_repr(self, operation: np.ndarray) -> np.ndarray:
        """Builds the permutation representation matrix for a
        given element of the group, applied on the cluster's sites.
        """
        T = np.zeros(shape=(len(self.positions), len(self.positions)))
        com_positions = self._get_center_of_mass()
        for i, atom in enumerate(com_positions):
            new_atom = operation @ atom
            for idx, pos in enumerate(com_positions):
                if np.allclose(new_atom, pos):
                    T[idx, i] = 1
                    break

        return T

    def project_onto_irrep(
        self,
        basis_vector: np.ndarray,
        irrep_name: str,
    ) -> np.ndarray:
        """Applies the projector method to given displacements according
        to one of the irreps.

        (Note: basis vector should be coherent w/ permutation repr.
        dimension is required for the expression of the displacement.)
        """
        extended_chars = np.repeat(
            self.character_table[irrep_name], self.classes_multiplicity
        )

        salc = np.zeros(len(self.positions), dtype=complex)
        for i, (_, op_matrix) in enumerate(self.operations.items()):
            P = self.build_permutation_repr(op_matrix)
            salc += extended_chars[i] * (P @ basis_vector)

        return salc

    def get_bath_parametrization(self) -> dict:
        """Returns the symmetry adapted coordinates in the orbital basis.

        Keys are irrep labels; values are lists of SALCs (one per orbit
        that projects non-trivially onto that irrep). Irreps with zero
        multiplicity in the permutation representation are excluded.
        """
        multiplicities = self._get_irrep_multiplicity_from_repr()
        contributing_irreps = [
            irrep
            for irrep, mj in zip(self.character_table, multiplicities)
            if not np.isclose(mj, 0)
        ]

        bath_parametrization = {}
        for irrep in contributing_irreps:
            salcs = []
            for k in range(len(self.positions)):
                seed = np.zeros(len(self.positions))
                seed[k] = 1
                v = self.project_onto_irrep(seed, irrep)
                if np.allclose(v, 0):
                    continue
                if salcs:
                    basis = np.column_stack(salcs)
                    residual = v - basis @ np.linalg.lstsq(basis, v, rcond=None)[0]
                    if np.allclose(residual, 0):
                        continue
                salcs.append(v)
            bath_parametrization[irrep] = salcs

        return bath_parametrization

    def get_pyqcm_generators(self, n_baths: int, abelian_pg: str) -> dict:
        """Return ready-to-use generators lists for pyqcm.cluster_model per irrep.

        Combines the cluster-site permutations and the bath-orbital phases into
        the format expected by pyqcm (bath_irrep=True):
          - cluster entries: 1-based site permutation indices (pyqcm subtracts 1)
          - bath entries: phase integers (multiples of 2π/|G|)

        Returns an empty dict for non-abelian point groups or groups without
        defined generators, signalling that bath_irrep symmetry cannot be used.

        Parameters
        ----------
        n_baths : int
            Number of bath orbitals per subbath.

        Returns
        -------
        dict
            {irrep_label: [[gen0_site_perm..., gen0_bath_phases...], [gen1...], ...]}
            Empty dict if the point group has no abelian generator structure.
        """
        pg = all_point_groups[abelian_pg]
        G = len(pg.operations)
        bath_param = self.get_bath_parametrization()

        # 1-based cluster-site permutation for each generator
        cluster_perms = {}
        for gen_name, gen_matrix in pg.generators.items():
            P = self.build_permutation_repr(gen_matrix)
            cluster_perms[gen_name] = (np.argmax(P, axis=0) + 1).tolist()

        # Bath phase for each (generator, irrep): 0 if symmetric, G//2 if antisymmetric
        bath_phases = {}
        for gen_name, gen_matrix in pg.generators.items():
            P = self.build_permutation_repr(gen_matrix)
            bath_phases[gen_name] = {}
            for irrep, salc_list in bath_param.items():
                coeffs = salc_list[0]
                projected = P @ coeffs
                bath_phases[gen_name][irrep] = (
                    0 if np.allclose(coeffs, projected) else G // 2
                )

        # Assemble per-irrep generator lists
        return {
            irrep: [
                cluster_perms[gen_name] + [bath_phases[gen_name][irrep]] * n_baths
                for gen_name in cluster_perms
            ]
            for irrep in bath_param
        }

    def show_parametrized_cluster(self, show: bool = True) -> Axes:
        """Build the cluster figure, one panel per SALC.
        """
        irrep_bath_parametrization = self.get_bath_parametrization()

        panels = []
        for irrep_label, salc_list in irrep_bath_parametrization.items():
            for k, salc in enumerate(salc_list):
                dim = len(salc_list)
                title = irrep_label if dim == 1 else f"{irrep_label} ({k + 1}/{dim})"
                panels.append((title, salc))

        fig, ax = plt.subplots(nrows=1, ncols=len(panels))
        if len(panels) == 1:
            ax = [ax]

        for axis, (title, coeffs) in zip(ax, panels):
            axis.set_title(title)
            for j, coeff in enumerate(coeffs):
                if coeff.imag > 0 and not np.isclose(coeff.imag, 0):
                    color = "go"
                elif coeff.imag < 0 and not np.isclose(coeff.imag, 0):
                    color = "bo"
                elif coeff.real > 0:
                    color = "ko"
                elif coeff.real < 0:
                    color = "ro"
                else:
                    axis.plot(
                        self.positions[j, 0],
                        self.positions[j, 1],
                        "o",
                        markerfacecolor="None",
                        markeredgecolor="black",
                        markersize=8,
                    )
                    continue
                axis.plot(
                    self.positions[j, 0], self.positions[j, 1], color, markersize=8
                )

            axis.set_xticks([])
            axis.set_yticks([])
            is_1d = np.allclose(self.positions[:, 1], self.positions[0, 1])
            if is_1d:
                axis.set_axis_off()
                axis.margins(x=0.2, y=0.2)
            else:
                axis.set_aspect("equal", adjustable="box")

        legend_entries = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="k",
                markersize=8,
                label="+1",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="r",
                markersize=8,
                label="−1",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="g",
                markersize=8,
                label="+i",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="b",
                markersize=8,
                label="−i",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",
                markeredgecolor="k",
                markersize=8,
                label="0",
            ),
        ]
        fig.legend(
            handles=legend_entries,
            loc="upper center",
            ncol=5,
            frameon=False,
            fontsize=9,
        )
        plt.subplots_adjust(bottom=0.1)

        if show:
            plt.tight_layout(rect=[0, 0, 1, 1])
            plt.show()

        return ax


if __name__ == "__main__":
    pass
