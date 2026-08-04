"""Parametrizing the bath automatically using character
table and generators of an abelian point group (Pyqcm formalism).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .point_groups import all_point_groups


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
        self.positions = np.asarray(site_positions, dtype=float)
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

    def _orbit_filtered_salcs(self, linked_sites: np.ndarray) -> dict:
        """Select the SALCs from get_bath_parametrization() whose support
        lies entirely within linked_sites. SALCs seeded from different
        orbits of the site-permutation representation are already
        linearly independent and orbit-localized, so this only selects
        among existing SALCs; it never rescales or zeroes coefficients.

        Survivors are labelled 'irrep' if only one orbit of that irrep
        survives, or 'irrep_1', 'irrep_2', ... if more than one does.

        Raises ValueError if a SALC's support partially overlaps
        linked_sites (linked_sites is not a union of point-group orbits),
        or if no SALC survives the filter.
        """
        linked = {int(i) for i in linked_sites}
        bath_param = self.get_bath_parametrization()

        survivors = {}
        for irrep, salc_list in bath_param.items():
            kept = []
            for salc in salc_list:
                support = {
                    i for i in range(len(salc)) if not np.isclose(abs(salc[i]), 0)
                }
                if support.issubset(linked):
                    kept.append(salc)
                elif support & linked:
                    raise ValueError(
                        f"linked_sites {sorted(linked)} partially overlaps "
                        f"the support {sorted(support)} of a '{irrep}' "
                        "SALC; linked_sites must be a union of point-group "
                        "orbits."
                    )
            if kept:
                survivors[irrep] = kept

        labelled = {}
        for irrep, kept in survivors.items():
            if len(kept) == 1:
                labelled[irrep] = kept[0]
            else:
                for k, salc in enumerate(kept):
                    labelled[f"{irrep}_{k + 1}"] = salc

        if not labelled:
            raise ValueError(
                f"No SALCs found with support inside linked_sites {sorted(linked)}."
            )

        return labelled

    def get_hybridization_links(
        self,
        nb: int,
        subbath: dict | None = None,
        linked_sites=None,
    ) -> dict:
        """Assign bath orbitals to SALCs, per subbath.

        Parameters
        ----------
        nb : int
            Number of bath orbitals per subbath.
        subbath : dict, optional
            {"nsb": number of subbaths (default 1),
             "irreps": "replica" (default), "unique", or "custom"/"mixed"}.
        linked_sites : sequence of int, optional
            0-based cluster site indices physically coupled to the bath.
            Defaults to all sites. Must be a union of point-group orbits
            (see _orbit_filtered_salcs).

        Returns
        -------
        dict
            {subbath_index (1-based int): {salc_label: {
                "coefficients": np.ndarray, "n_orbitals": int}}}

        Raises
        ------
        ValueError
            If linked_sites is not a union of orbits, if no SALC survives
            the filter, if nb/nsb fail the mode's divisibility
            requirement, or if subbath["irreps"] is not recognized.
        NotImplementedError
            If subbath["irreps"] is "custom" or "mixed".
        """
        subbath = subbath or {}
        nsb = subbath.get("nsb", 1)
        mode = subbath.get("irreps", "replica")

        if linked_sites is None:
            linked_sites = range(len(self.positions))

        salcs = self._orbit_filtered_salcs(np.asarray(list(linked_sites)))
        n_units = len(salcs)

        if mode == "replica":
            if nb % (2 * n_units) != 0:
                raise ValueError(
                    f"'replica' mode requires nb ({nb}) to be divisible "
                    f"by 2 * n_units ({2 * n_units})."
                )
            n_orbitals = nb // n_units
            subbath_links = {
                label: {"coefficients": salc, "n_orbitals": n_orbitals}
                for label, salc in salcs.items()
            }
            return {sb: dict(subbath_links) for sb in range(1, nsb + 1)}

        if mode == "unique":
            if nsb % n_units != 0:
                raise ValueError(
                    f"'unique' mode requires nsb ({nsb}) to be divisible "
                    f"by n_units ({n_units})."
                )
            if nb % 2 != 0:
                raise ValueError(f"'unique' mode requires nb ({nb}) to be even.")
            n = nsb // n_units
            labels = list(salcs)
            return {
                sb: {
                    labels[(sb - 1) // n]: {
                        "coefficients": salcs[labels[(sb - 1) // n]],
                        "n_orbitals": nb,
                    }
                }
                for sb in range(1, nsb + 1)
            }

        if mode in ("custom", "mixed"):
            raise NotImplementedError(
                "custom subbath assignment is not yet implemented"
            )

        raise ValueError(f"Unknown subbath['irreps'] mode: {mode!r}")

    def get_pyqcm_generators(
        self,
        n_baths: int,
        abelian_pg: str,
        subbath: dict | None = None,
        linked_sites=None,
    ) -> list | dict:
        """Return ready-to-use pyqcm.cluster_model generators (bath_irrep=True
        format): cluster-site permutations combined with bath-orbital phases.

        The bath-orbital-per-SALC split is delegated to get_hybridization_links,
        so this accepts the same subbath/linked_sites arguments. For each
        point-group generator, a row is built as:
          cluster site permutation (1-based) + bath phases, one block per SALC
          label of the subbath, each phase repeated n_orbitals times.
        Block order follows the SALC label order returned by
        get_hybridization_links (character-table order, or 'irrep_1',
        'irrep_2', ... when an irrep has multiplicity > 1) -- declare
        eb{i}/tb{i} bath parameters in that same order.

        Parameters
        ----------
        n_baths : int
            Total number of bath orbitals (see get_hybridization_links).
        abelian_pg : str
            Point group supplying the generators (its .generators dict).
        subbath : dict, optional
            Forwarded to get_hybridization_links. Default (nsb=1, 'replica')
            is the vanilla, non-SB-CDMFT mixed bath.
        linked_sites : sequence of int, optional
            Forwarded to get_hybridization_links.

        Returns
        -------
        list or dict
            A flat generators list -- [[gen0_site_perm..., gen0_bath_phases...],
            [gen1...], ...] -- ready for cluster_model(generators=result,
            bath_irrep=True), when subbath['nsb'] is 1 (the default). When
            subbath['nsb'] > 1, {subbath_index: generators_list, ...} instead,
            one flat list per subbath cluster_model.
            Empty dict if abelian_pg has no defined generators (non-abelian or
            unsupported point group), signalling that bath_irrep symmetry
            cannot be used.
        """
        pg = all_point_groups[abelian_pg]
        if not pg.generators:
            return {}

        G = len(pg.operations)
        links = self.get_hybridization_links(
            n_baths, subbath=subbath, linked_sites=linked_sites
        )

        # 1-based cluster-site permutation, and permutation matrix, per generator
        cluster_perms = {}
        generator_matrices = {}
        for gen_name, gen_matrix in pg.generators.items():
            generator_matrices[gen_name] = self.build_permutation_repr(gen_matrix)
            cluster_perms[gen_name] = (
                np.argmax(generator_matrices[gen_name], axis=0) + 1
            ).tolist()

        def phase(gen_name: str, coeffs: np.ndarray) -> int:
            """0 if coeffs is symmetric under gen_name, G//2 if antisymmetric."""
            projected = generator_matrices[gen_name] @ coeffs
            return 0 if np.allclose(coeffs, projected) else G // 2

        def row(gen_name: str, salc_labels: dict) -> list:
            generator_row = list(cluster_perms[gen_name])
            for info in salc_labels.values():
                generator_row.extend(
                    [phase(gen_name, info["coefficients"])] * info["n_orbitals"]
                )
            return generator_row

        generators_by_subbath = {
            sb: [row(gen_name, salc_labels) for gen_name in cluster_perms]
            for sb, salc_labels in links.items()
        }

        nsb = (subbath or {}).get("nsb", 1)
        if nsb == 1:
            return generators_by_subbath[1]
        return generators_by_subbath

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
