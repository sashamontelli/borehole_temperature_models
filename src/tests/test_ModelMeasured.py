import pickle

from pathlib import Path

import numpy as np
import pytest

from borehole_temperature_models.ModelMeasured import ModelMeasured


# ----------------------------------------------------------------------
def _GetMeasuredDataFilenames() -> list[str]:
    data_dir = Path(__file__).parent / "data" / "ModelMeasured"
    assert data_dir.is_dir(), data_dir

    data_filenames: list[str] = []

    for data_filename in data_dir.iterdir():
        if not data_filename.name.startswith("measured"):
            continue

        data_filenames.append(str(data_filename))

    return data_filenames


@pytest.mark.parametrize("data_filename", _GetMeasuredDataFilenames())
def test_ModelMeasured(data_filename):
        with Path(data_filename).open("rb") as f:
            theta, z, output = pickle.load(f)

        generated_output = ModelMeasured(
            *theta,
            z=z,
        )

        assert np.array_equal(generated_output, output)
