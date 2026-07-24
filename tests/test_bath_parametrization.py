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
