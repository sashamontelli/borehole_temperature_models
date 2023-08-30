import copy
import json
import pickle
import re

from pathlib import Path
from typing import Iterator

import numpy as np
import numpy.typing as npt
import pytest

from icetemp.TemperatureModel import TemperatureModel


# ----------------------------------------------------------------------
_data_dir                                   = Path(__file__).parent / "data"


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def test_Create1() -> None:
    model = TemperatureModel.Create(1, [360,470], 500, 10, 5e4, 1e4, 1e4, 7e2, [-24.3, -24.3], [0.11, 0.11], 950e-4)

    last_known_good = TemperatureModel.Load(_data_dir / "TemperatureModel" / "Create1.json")

    assert TemperatureModel.Compare(
        model,
        last_known_good,
        allow_tolerance=True,
    )


# ----------------------------------------------------------------------
def test_Create2() -> None:
    model = TemperatureModel.Create(1, [400,470], 500, 10, 5e4, 1e4, 1e4, 2e3, [-30, -25], [0.2, 0.1], 700e-4)

    last_known_good = TemperatureModel.Load(_data_dir / "TemperatureModel" / "Create2.json")

    assert TemperatureModel.Compare(
        model,
        last_known_good,
        allow_tolerance=True,
    )


# ----------------------------------------------------------------------
def _GetModelDataFilenames() -> list[str]:
    data_dir = _data_dir / "TemperatureModel"
    assert data_dir.is_dir(), data_dir

    data_filenames: list[str] = []

    for data_filename in data_dir.iterdir():
        if not data_filename.name.startswith("model"):
            continue

        data_filenames.append(str(data_filename))

    return data_filenames


@pytest.mark.parametrize("data_filename", _GetModelDataFilenames())
def test_ModelDataFile(
    data_filename: str,
    benchmark,
):
    with Path(data_filename).open("rb") as f:
        content = pickle.load(f)

    # `dt_years` was converted to a parameter after all of the data was serialized. Fortunately,
    # the data was a serialized with a value of 1. Add it here now.
    content["input"]["dt_years"] = 1

    original_model = TemperatureModel(
        content["output"]["Tmeasured"],
        content["output"]["Tvar_H_Tmatrix"],
        content["output"]["zvar_H_Tmatrix"],
    )

    created_model = benchmark(TemperatureModel.Create, **content["input"])

    assert TemperatureModel.Compare(
        created_model,
        original_model,
        allow_tolerance=True,
    )


# ----------------------------------------------------------------------
@pytest.mark.parametrize("attribute_name", ["Tmeasured", "Tvar_H_Tmatrix", "zvar_H_Tmatrix"])
def test_NaN(
    attribute_name: str,
) -> None:
    d: dict[str, npt.NDArray[np.float64]] = {
        "Tmeasured": np.random.rand(3, 2),
        "Tvar_H_Tmatrix": np.random.rand(3, 2),
        "zvar_H_Tmatrix": np.random.rand(3, 2),
    }

    d[attribute_name][0, 1] = float("nan")

    with pytest.raises(
        ValueError,
        match=re.escape("A NaN was encountered in '{}' at (0, 1).".format(attribute_name)),
    ):
        TemperatureModel(**d)


# ----------------------------------------------------------------------
def test_SaveAndLoad(fs, _random_model) -> None:
    filename = Path("test")

    _random_model.Save(filename)
    loaded_model = TemperatureModel.Load(filename)

    assert loaded_model == _random_model


# ----------------------------------------------------------------------
def test_LoadInvalidFilename(fs) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("The file 'does_not_exist' does not exist."),
    ):
        TemperatureModel.Load(Path("does_not_exist"))


# ----------------------------------------------------------------------
def test_LoadInvalidVersion(fs, _random_model) -> None:
    filename = Path("test")

    _random_model.Save(filename)

    with filename.open() as f:
        content = json.load(f)

    invalid_version = "9999999.0.0"

    content["version"] = invalid_version

    with filename.open("w") as f:
        json.dump(content, f)

    with pytest.raises(
        Exception,
        match=re.escape("'{}' is not a supported version.".format(invalid_version)),
    ):
        TemperatureModel.Load(filename)


# ----------------------------------------------------------------------
def test_LoadMissingData(fs, _random_model) -> None:
    filename = Path("test")

    _random_model.Save(filename)

    with filename.open() as f:
        content = json.load(f)

    del content["version"]

    with filename.open("w") as f:
        json.dump(content, f)

    with pytest.raises(
        Exception,
        match=re.escape("The content in 'test' appears to be corrupt; 'version' was not found."),
    ):
        TemperatureModel.Load(filename)


# ----------------------------------------------------------------------`
@pytest.mark.parametrize("attribute_name", ["Tmeasured", "Tvar_H_Tmatrix", "zvar_H_Tmatrix"])
def test_CompareMethods(
    _random_model: TemperatureModel,
    attribute_name: str,
) -> None:
    assert TemperatureModel.Compare(_random_model, _random_model) is True
    assert _random_model == _random_model
    assert (_random_model != _random_model) is False

    original_model = copy.deepcopy(_random_model)

    value = getattr(_random_model, attribute_name)

    value[0, 0] += 1

    assert TemperatureModel.Compare(_random_model, original_model) is False
    assert (_random_model == original_model) is False
    assert _random_model != original_model


# ----------------------------------------------------------------------
def test_CompareDifferentTypes(_random_model):
    assert (_random_model == 3) is False
    assert (_random_model != 3) is True


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
@pytest.fixture
def _random_model() -> Iterator[TemperatureModel]:
    yield TemperatureModel(
        np.random.rand(3, 2),
        np.random.rand(1, 3),
        np.random.rand(3, 1),
    )
