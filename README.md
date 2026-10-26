# meristem

meristem[^name] is a Python toolkit for building and running parameter ensembles with the Community Land Model (CLM) and CLM-FATES.
It turns a spreadsheet of parameter ranges into ready-to-run parameter files, using Latin Hypercube, one-at-a-time or Morris
sampling. It can also chain the stages of a model run, such as spinup and transient, into PBS jobs that submit themselves one after another.

## Scope

meristem is developed for CLM and CLM-FATES, but its core is model-agnostic:

- **Sampling** (Latin Hypercube, one-at-a-time, Morris, parameter groups, posterior draws) works on normalized
  parameter values in [0, 1] and knows nothing about the model.
- **Parameter files** are read and written through a small `ParameterDataset` interface. NetCDF (CLM) and
  FATES JSON are supported today.
- **Job chaining** runs each stage's setup script and submits the next stage on success. Stages built on
  CIME cases are supported directly, and stages with `no_cime: true` can run any script.

Contributing support for another model may involve adding a `ParameterDataset` subclass for its parameter-file
format and, for job chaining, setup scripts for its run stages. Job chaining currently requires a PBS scheduler.

## Workflow

meristem has three parts:

1. **`meristem.paramgen`**: generate parameter files. Describe parameters and their ranges in a CSV,
   choose a sampling design, and get one parameter file per ensemble member, plus a key of the
   sampled values. This runs anywhere.
2. **`meristem.run`**: run one model run. Describe its stages (e.g. spinup → transient) in a YAML
   file, and submit them as a chain of PBS jobs that advance on success and stop cleanly on failure.
   A run can also be a single stage. This runs on a PBS cluster with CESM/CTSM.
3. **`meristem.ensemble`** *(in progress)*: run an ensemble. One run per paramgen parameter file,
   each with one or more stages, submitted and tracked together.

`paramgen` and `run` can each be used on their own. `ensemble` builds on both.

## Installation

meristem requires Python 3.12 or newer. It is not yet on PyPI or conda-forge, so install it from source.
The recommended way is the provided conda environment:

```bash
git clone https://github.com/adrifoster/meristem.git
cd meristem
conda env create -f environment.yml
conda activate meristem
```

This installs meristem in editable mode, along with its dependencies and the development tools
(pytest, pylint, black).

To install into an existing environment instead, run `pip install -e .` from the repository root
(or `pip install -e ".[dev]"` to include the development tools).

## Documentation

- [docs/paramgen.md](docs/paramgen.md): parameter ensembles in detail: the `main.csv` format,
  Latin Hypercube, one-at-a-time and Morris designs, parameter types, grouping, fixed indices
  and posterior sampling.
- [docs/run.md](docs/run.md): runs in detail: how the tethered PBS jobs fit together,
  the config reference, writing setup scripts, and the `meristem run` command reference.
- [examples/paramgen](examples/paramgen): worked CLM and FATES examples, including posterior sampling.

## Quick start

### Generate a parameter file ensemble

The repository includes a small CLM example. From the repository root:

```bash
cd examples/paramgen/clm_example
meristem paramgen lh_config.yaml
```

`lh_config.yaml` asks for a 20-member Latin Hypercube ensemble:

```yaml
ensemble_type: LatinHypercube
param_dir: clm_param_dir              # main.csv (+ per-PFT CSVs) describing the parameters to vary
default_param_file: default_clm_param.nc
ensemble_dir: output_LH
file_prefix: clm_lh
ensemble_members: 20
```

This writes `output_LH/clm_lh_0000.nc` through `clm_lh_0019.nc`, a key file `clm_lh_key.csv` recording
the normalized value each parameter took in each member, and `clm_lh.txt` listing the member names.

The same thing from Python:

```python
from meristem.paramgen import ParamEnsemble

ensemble = ParamEnsemble.from_dict({
    "ensemble_type": "LatinHypercube",
    "param_dir": "clm_param_dir",
    "default_param_file": "default_clm_param.nc",
    "ensemble_dir": "output_LH",
    "file_prefix": "clm_lh",
    "ensemble_members": 20,
})
ensemble.create_ensemble()
```

### Chain the stages of a model run

Write a `run.yaml` listing your simulation stages (e.g., spinup, historical, transient) and the setup script for each one (a script that creates, configures and builds the case, but does not submit it):

```yaml
root: /glade/derecho/scratch/user/my_spinup
run_id: my_spinup
project: PXXXXXXX

stages:
  - name: spinup_ad
    script: /path/to/setup_ad.sh
    queue: develop

  - name: spinup_postad
    script: /path/to/setup_postad.sh
    queue: develop
```

Then create the run, submit the first stage, and check on it:

```bash
meristem run create --config run.yaml
meristem run submit --root /glade/derecho/scratch/user/my_spinup
meristem run status --root /glade/derecho/scratch/user/my_spinup
```

Once each stage of a model run finishes successfully, the next stage is automatically submitted.

## Contributing

meristem is in early development and the API may change. Bug reports and feature requests are welcome as
[GitHub issues](https://github.com/adrifoster/meristem/issues). If you'd like to contribute code, please
open an issue first to discuss the change.

To run the tests and linters:

```bash
pytest                       # unit tests (PBS tests are skipped by default)
pylint src/meristem
black --check src tests
```

## License

meristem is licensed under the [Apache License 2.0](LICENSE).

[^name]: A meristem is the growing tissue in plants that gives rise to new organs.
