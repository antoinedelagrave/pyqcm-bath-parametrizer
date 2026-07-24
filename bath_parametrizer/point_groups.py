"""This modules contains information about the considered
point groups for SB-CDMFT simulations, i.e. the character
tables and groups' operations. Convention for the character
tables from Bilbao Crystallographic Server (BCS).
"""

import numpy as np


class E:
    def __init__(self) -> None:
        self.is_abelian = True
        self.operations = {
            "E": np.eye(3),
        }
        self.generators = {}
        self.character_table = {
            "A": np.array([1]),
        }
        self.classes_multiplicity = np.array([1])
        return


E_point_group = E()

# ----------------------------------------------------------------------------


class Cs:
    def __init__(self) -> None:
        self.is_abelian = True
        self.operations = {
            "E": np.eye(3),
            "Cs": np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
        }
        self.generators = {}
        self.character_table = {
            "A": np.array([1, 1]),
            "A'": np.array([1, -1]),
        }
        self.classes_multiplicity = np.array([1, 1])
        return


Cs_point_group = Cs()

# ----------------------------------------------------------------------------


class C2:
    def __init__(self) -> None:
        self.is_abelian = True
        self.operations = {
            "E": np.eye(3),
            "C2": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
        }
        self.generators = {
            "C2": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
        }
        self.character_table = {
            "A": np.array([1, 1]),
            "B": np.array([1, -1]),
        }
        self.classes_multiplicity = np.array([1, 1])
        return


C2_point_group = C2()


# ----------------------------------------------------------------------------


class C2v:
    def __init__(self) -> None:
        self.is_abelian = True
        self.operations = {
            "E": np.eye(3),
            "C2": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            "sigma_y": np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            "sigma_x": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        }
        self.generators = {
            "C2": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            "sigma_y": np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
        }
        self.character_table = {
            "A1": np.array([1, 1, 1, 1]),
            "A2": np.array([1, 1, -1, -1]),
            "B1": np.array([1, -1, 1, -1]),
            "B2": np.array([1, -1, -1, 1]),
        }
        self.classes_multiplicity = np.array([1, 1, 1, 1])
        return


C2v_point_group = C2v()

# ----------------------------------------------------------------------------


class C3:
    def __init__(self) -> None:
        self.is_abelian = False
        self.operations = {
            "E": np.eye(3),
            "C3+": np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]]),
            "C3-": np.array([[-1, 1, 0], [-1, 0, 0], [0, 0, 1]]),
        }
        self.generators = {}

        phase = np.exp(2j * np.pi / 3)
        self.character_table = {
            "A": np.array([1, 1, 1]),
            "E1": np.array([1, phase**2, phase]),
            "E2": np.array([1, phase, phase**2]),
        }
        self.classes_multiplicity = np.array([1, 1, 1])
        return


C3_point_group = C3()

# ----------------------------------------------------------------------------


class C3v:
    def __init__(self) -> None:
        self.is_abelian = False
        self.operations = {
            "E": np.eye(3),
            "C3+": np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]]),
            "C3-": np.array([[-1, 1, 0], [-1, 0, 0], [0, 0, 1]]),
            "sigma_d1": np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]),
            "sigma_x": np.array([[-1, 1, 0], [0, 1, 0], [0, 0, 1]]),
            "sigma_y": np.array([[1, 0, 0], [1, -1, 0], [0, 0, 1]]),
        }
        self.generators = {}

        self.character_table = {
            "A1": np.array([1, 1, 1]),
            "A2": np.array([1, 1, -1]),
            "E": np.array([2, -1, 0]),
        }
        self.classes_multiplicity = np.array([1, 2, 3])
        return


C3v_point_group = C3v()

# ----------------------------------------------------------------------------


class C4:
    def __init__(self) -> None:
        self.is_abelian = False
        self.operations = {
            "E": np.eye(3),
            "C2": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            "C4+": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            "C4-": np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
        }
        self.generators = {}
        self.character_table = {
            "A": np.array([1, 1, 1, 1]),
            "B": np.array([1, 1, -1, -1]),
            "E1": np.array([1, -1, -1j, 1j]),
            "E2": np.array([1, -1, 1j, -1j]),
        }
        self.classes_multiplicity = np.array([1, 1, 1, 1])
        return


C4_point_group = C4()

# ----------------------------------------------------------------------------


class C4v:
    def __init__(self) -> None:
        self.is_abelian = False
        self.operations = {
            "E": np.eye(3),
            "C2": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            "C4+": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            "C4-": np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
            "sigma_y": np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            "sigma_x": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            "sigma_d1": np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]),
            "sigma_d2": np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
        }
        self.generators = {}
        self.character_table = {
            "A1": np.array([1, 1, 1, 1, 1]),
            "A2": np.array([1, 1, 1, -1, -1]),
            "B1": np.array([1, 1, -1, 1, -1]),
            "B2": np.array([1, 1, -1, -1, 1]),
            "E": np.array([2, -2, 0, 0, 0]),
        }
        self.classes_multiplicity = np.array([1, 1, 2, 2, 2])
        return


C4v_point_group = C4v()

# ----------------------------------------------------------------------------


class C6:
    def __init__(self) -> None:
        self.is_abelian = False
        self.operations = {
            "E": np.eye(3),
            "C3+": np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]]),
            "C3-": np.array([[-1, 1, 0], [-1, 0, 0], [0, 0, 1]]),
            "C2": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            "C6-": np.array([[0, 1, 0], [-1, 1, 0], [0, 0, 1]]),
            "C6+": np.array([[1, -1, 0], [1, 0, 0], [0, 0, 1]]),
        }
        self.generators = {}

        phase = np.exp(2j * np.pi / 3)
        self.character_table = {
            "A": np.array([1, 1, 1, 1, 1, 1]),
            "B": np.array([1, -1, 1, -1, 1, -1]),
            "1E2": np.array([1, phase, phase**2, 1, phase, phase**2]),
            "2E2": np.array([1, phase**2, phase, 1, phase**2, phase]),
            "2E1": np.array([1, -(phase**2), phase, -1, phase**2, -phase]),
            "1E1": np.array([1, -phase, phase**2, -1, phase, -(phase**2)]),
        }
        self.classes_multiplicity = np.array([1, 1, 1, 1, 1, 1])
        return


C6_point_group = C6()

# ----------------------------------------------------------------------------


class C6v:
    def __init__(self) -> None:
        self.is_abelian = False
        self.operations = {
            "E": np.eye(3),
            "C3+": np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]]),
            "C3-": np.array([[-1, 1, 0], [-1, 0, 0], [0, 0, 1]]),
            "C2": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            "C6-": np.array([[0, 1, 0], [-1, 1, 0], [0, 0, 1]]),
            "C6+": np.array([[1, -1, 0], [1, 0, 0], [0, 0, 1]]),
            "sigma_d1": np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]),
            "sigma_x": np.array([[-1, 1, 0], [0, 1, 0], [0, 0, 1]]),
            "sigma_d2": np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
            "sigma_x120": np.array([[1, -1, 0], [0, -1, 0], [0, 0, 1]]),
            "sigma_d220": np.array([[-1, 0, 0], [-1, 1, 0], [0, 0, 1]]),
        }
        self.generators = {}
        self.character_table = {
            "A1": np.array([1, 1, 1, 1, 1, 1]),
            "A2": np.array([1, 1, 1, 1, -1, -1]),
            "B1": np.array([1, -1, 1, -1, -1, 1]),
            "B2": np.array([1, -1, 1, -1, 1, -1]),
            "E1": np.array([2, -2, -1, 1, 0, 0]),
            "E2": np.array([2, 2, -1, -1, 0, 0]),
        }
        self.classes_multiplicity = np.array([1, 1, 2, 2, 3, 3])
        return


C6v_point_group = C6v()

# ----------------------------------------------------------------------------

all_point_groups = {
    "E": E_point_group,
    "Cs": Cs_point_group,
    "C2": C2_point_group,
    "C2v": C2v_point_group,
    "C3": C3_point_group,
    "C3v": C3v_point_group,
    "C4": C4_point_group,
    "C4v": C4v_point_group,
    "C6": C6_point_group,
    "C6v": C6v_point_group,
}

if __name__ == "__main__":
    pass
