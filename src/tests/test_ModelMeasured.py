import pickle

from pathlib import Path

import numpy as np
import pytest

from icetemp.ModelMeasured import ModelMeasured


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

        # `dt_years` was converted to a parameter after all of the data was serialized. Fortunately,
        # the data was a serialized with a value of 1. Add it here now.
        theta = np.insert(
            theta,
            0,      # index
            1,      # years
        )

        generated_output = ModelMeasured(*theta)

        assert np.array_equal(generated_output, output)
