import numpy as np
import pytest

from bath_parametrizer import bath_parametrization


class TestHybridizationLinksReplica:
    def setup_method(self):
        atoms = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float
        )
        self.parametrizer = bath_parametrization.BathParametrizer(atoms, "C2v")

    def test_default_is_one_replica_subbath_with_all_irreps(self):
        links = self.parametrizer.get_hybridization_links(8)
        assert set(links) == {1}
        assert set(links[1]) == {"A1", "A2", "B1", "B2"}
        for info in links[1].values():
            assert info["n_orbitals"] == 2

    def test_replica_multiple_subbaths_are_identical(self):
        links = self.parametrizer.get_hybridization_links(
            8, subbath={"nsb": 3, "irreps": "replica"}
        )
        assert set(links) == {1, 2, 3}
        for sb in (1, 2, 3):
            assert set(links[sb]) == {"A1", "A2", "B1", "B2"}
            for label in links[sb]:
                np.testing.assert_array_equal(
                    links[sb][label]["coefficients"],
                    links[1][label]["coefficients"],
                )

    def test_replica_bad_divisibility_raises_value_error(self):
        with pytest.raises(ValueError):
            self.parametrizer.get_hybridization_links(6)


class TestHybridizationLinksUnique:
    def setup_method(self):
        atoms = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
        self.parametrizer = bath_parametrization.BathParametrizer(atoms, "C2")

    def test_unique_grouped_assignment(self):
        links = self.parametrizer.get_hybridization_links(
            4, subbath={"nsb": 4, "irreps": "unique"}, linked_sites=[0, 3]
        )
        labels_in_order = [next(iter(links[sb])) for sb in (1, 2, 3, 4)]
        assert labels_in_order[0] == labels_in_order[1]
        assert labels_in_order[2] == labels_in_order[3]
        assert labels_in_order[0] != labels_in_order[2]
        for sb in (1, 2, 3, 4):
            label = labels_in_order[sb - 1]
            assert links[sb][label]["n_orbitals"] == 4

    def test_unique_bad_nsb_divisibility_raises_value_error(self):
        with pytest.raises(ValueError):
            self.parametrizer.get_hybridization_links(
                4, subbath={"nsb": 3, "irreps": "unique"}, linked_sites=[0, 3]
            )

    def test_unique_odd_nb_raises_value_error(self):
        with pytest.raises(ValueError):
            self.parametrizer.get_hybridization_links(
                3, subbath={"nsb": 2, "irreps": "unique"}, linked_sites=[0, 3]
            )


class TestHybridizationLinksCustomAndUnknownModes:
    def setup_method(self):
        atoms = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
        self.parametrizer = bath_parametrization.BathParametrizer(atoms, "C2")

    def test_mixed_mode_raises_not_implemented_error(self):
        with pytest.raises(NotImplementedError):
            self.parametrizer.get_hybridization_links(
                4, subbath={"nsb": 1, "irreps": "mixed"}
            )


class TestPyqcmGeneratorsMixedPlaquette:
    """2x2 plaquette (C2v) with 8 bath orbitals split 2-per-irrep across
    A1/A2/B1/B2. The standard vanilla (non-SB-CDMFT) mixed-bath CDMFT
    setup for this cluster.
    """

    def setup_method(self):
        atoms = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float
        )
        self.parametrizer = bath_parametrization.BathParametrizer(atoms, "C2v")

    def test_default_returns_flat_list_for_vanilla_mixed_bath(self):
        generators = self.parametrizer.get_pyqcm_generators(8, "C2v")
        assert isinstance(generators, list)
        assert len(generators) == 2
        for row in generators:
            assert len(row) == 4 + 8
            cluster_part = row[:4]
            assert sorted(cluster_part) == [1, 2, 3, 4]

    def test_mixed_bath_phase_pattern_matches_c2v_characters(self):
        c2_row, sigma_y_row = self.parametrizer.get_pyqcm_generators(8, "C2v")
        assert c2_row[4:] == [0, 0, 0, 0, 2, 2, 2, 2]
        assert sigma_y_row[4:] == [0, 0, 2, 2, 0, 0, 2, 2]

    def test_nsb_greater_than_one_returns_dict_of_identical_flat_lists(self):
        generators = self.parametrizer.get_pyqcm_generators(
            8, "C2v", subbath={"nsb": 3, "irreps": "replica"}
        )
        assert set(generators) == {1, 2, 3}
        for sb in (1, 2, 3):
            assert generators[sb] == generators[1]

    def test_unique_mode_passthrough_single_irrep_per_subbath(self):
        generators = self.parametrizer.get_pyqcm_generators(
            4, "C2v", subbath={"nsb": 4, "irreps": "unique"}
        )
        assert set(generators) == {1, 2, 3, 4}
        for row_list in generators.values():
            for row in row_list:
                bath_part = row[4:]
                assert len(bath_part) == 4
                assert len(set(bath_part)) == 1

    def test_no_generators_returns_empty_dict(self):
        assert self.parametrizer.get_pyqcm_generators(4, "C4") == {}


class TestOrbitFilteredSalcs:
    def setup_method(self):
        atoms = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
        self.parametrizer = bath_parametrization.BathParametrizer(atoms, "C2")

    def test_keeps_only_salcs_supported_on_linked_sites(self):
        salcs = self.parametrizer._orbit_filtered_salcs(np.array([0, 3]))
        assert set(salcs) == {"A", "B"}
        for salc in salcs.values():
            support = {i for i in range(4) if not np.isclose(abs(salc[i]), 0)}
            assert support == {0, 3}

    def test_partial_overlap_raises_value_error(self):
        with pytest.raises(ValueError):
            self.parametrizer._orbit_filtered_salcs(np.array([0, 1]))

    def test_empty_result_raises_value_error(self):
        with pytest.raises(ValueError):
            self.parametrizer._orbit_filtered_salcs(np.array([], dtype=int))
