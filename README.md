# Englacial borehole temperature model

Code to solve the one-dimensional englacial temperature evolution model (both inversion and forward sensitivity simulations) presented in the article *Geothermal heat flux is the dominant source of uncertainty in englacial-temperature-based dating of ice-rise formation* by Montelli and Kingslake in The Cryosphere.

The python code needed to run the model is included. Some of the figures presented in the paper were further tidied up; the resulting rasters are provided in the directory *figures*.

All code runs in Python 3.9 and usually would require some modules to be installed before they can be imported.

---

### Downloading and running the code

The easiest way to run this code is via Jupyter Notebook.

#### Running the python code locally

To install the python dependencies required by this code, run `pip install -e .`.

#### Developing the code locally

1. `pip install -e ".[dev]"`
1. `python Build.py Test`

#### Building the package

To build a package/wheel for distribution, run:

1. `pip install -e ".[dev,package]"`
1. `python Build.py UpdateVersion` [optional]
1. `python Build.py Package`
1. `python Build.py Publish` [optional]

#### Improvements

Please let us know if you have any questions or suggestions through issues.

---

### Historical Naming (`icetemp` vs. `borehole_temperature_model`)

The name of python package produced by this repository is `icetemp` while the name of the repository is `borehole_temperature_model`.

The name `icetemp` was introduced when refactoring the code for distribution as a python package, with the goal
of making the algorithms more accessible to other researchers. It is our hope that the name `icetemp` more clearly illustrates the functionality provided by the package and makes it easier to discover on [PyPi](https://pypi.org/search/?q=icetemp).

---

### Acknowledgements

Thank you to Thorsten Albrecht for sharing the PISM model outputs.

Thank you to Nicholas Holschuh for sharing the digitised Crary Ice Rise borehole temperature measurements.

Thank you to the Schmidt Science Fellowship for providing funding and the opportunity to do this project.
