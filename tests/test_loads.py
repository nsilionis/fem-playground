"""Load vector: beam DOF names, wrong lengths, distributed loads."""

import numpy as np
import pytest

from femlib.assembly import assemble_load_vector
from femlib.elements import Beam
from femlib.mesh import create_beam_mesh


def test_beam_tip_force_and_moment_rows():
    mesh = create_beam_mesh(2.0, 2)
    F = assemble_load_vector(
        mesh, {"concentrated": [{"location": (2.0,), "force": [-1e3, 5e2]}]},
        dof_names=Beam.dof_names,
    )
    expected = np.zeros(6)
    expected[4], expected[5] = -1e3, 5e2
    assert np.array_equal(F, expected)


def test_force_of_wrong_length_rejected():
    mesh = create_beam_mesh(2.0, 2)
    with pytest.raises(ValueError, match="one value per DOF"):
        assemble_load_vector(
            mesh, {"concentrated": [{"location": (2.0,), "force": [1.0]}]},
            dof_names=Beam.dof_names,
        )


def test_distributed_load_refused_for_beam():
    mesh = create_beam_mesh(2.0, 2)
    with pytest.raises(ValueError, match="Q4 meshes only"):
        assemble_load_vector(
            mesh,
            {"distributed": [{"edge": "top", "load_density": [0.0, -1.0]}]},
            dof_names=Beam.dof_names,
        )
