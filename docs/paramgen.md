# paramgen: generating parameter ensembles

`meristem.paramgen` generates parameter ensembles for CLM and CLM-FATES. Given a set of
parameters and their ranges (or posterior files), it produces a collection of parameter
files (NetCDF or FATES JSON) that can be fed directly into a set of CLM/FATES model runs.

## What it does

This library automates the process of creating different parameter sets to run in an
ensemble of model runs, the purpose of which can be calibration, sensitivity-testing,
or otherwise. To do this, the user must provide:

* A default FATES or CLM parameter file (NetCDF or JSON)
* A spreadsheet describing which parameters to vary and their allowed ranges (plus PFT-specific sheets if needed)
* An optional set of posterior text files for drawing from a posterior sample, plus a config file.
* A configuration file or dictionary specifying ensemble features

The library then produces one parameter file per ensemble member, plus a key file that
records what value each parameter took in each member, and a list of generated parameter files.

Three sampling strategies are supported:

1. Latin Hypercube: Generates `n` ensemble members by sampling the parameter space
evenly, ensuring each parameter's range is well-covered across the ensemble.
2. One-at-a-time (OAT): varies one parameter at a time between its minimum and maximum, while
holding all others at default (i.e., two ensemble members per parameter).
3. Morris: generates trajectories through parameter space that change one parameter at a
time, for Morris (elementary effects) global sensitivity analysis.


## Input files

### 1. Default parameter file

A standard FATES or CLM parameter file that will serve as the base for all ensemble members. Both NetCDF (.nc) and FATES JSON (.json) formats are supported. All parameters not being varied will keep their values from this file. This file is also used to grab default values for parameters that require a default for scaling (e.g., scaling by a percent).

### 2. Parameter directory

A folder containing CSV files that describe which parameters to vary:

* `main.csv` - one row per parameter, with the columns described below
* Optionally, one additional CSV per parameter with PFT-specific bounds (e.g. `fates_leaf_slatop.csv`)

**Required columns in `main.csv`:**

| Column | Description |
| --- | --- |
| `parameter_name` | Name of the parameter (usually the same as it appears in the NetCDF dataset) |
| `coord` | Dimensions of this parameter, e.g. `['fates_pft']` or `[]` for scalars |
| `param_type` | One of `default`, `sliced`, `scale_from_root`, or `joint` |
| `strategy` | Sampling strategy: `uniform` or `posterior` |
| `param_min` | Minimum value (a scalar value, "Xpercent" for percent-based bounds, or "pft" for PFT-specific bounds) |
| `param_max` | Maximum value (same options as `param_min`) |

**Optional columns:**

| Column | Description |
| --- | --- |
| `expand_dim` | If set, generate one independent parameter per index along this dimension (e.g. `fates_pft` to vary each PFT separately) |
| `group_name` | If set, all parameters sharing the same group name receive the same normalized sample value in each ensemble member |
| `slice_dim` | For `sliced` type: which dimension to index into |
| `slice_index` | For `sliced` type: which index along `slice_dim` to target |
| `base_params` | For `sliced`, `scale_from_root`, and `joint` types: the underlying NetCDF variable(s) being modified |
| `root_param` | For `scale_from_root` type: the parameter to scale from |
| `precision` | Optional rounding precision for output values, e.g. `.4f` rounds to 4 decimal places |

### 3. (*optional*) Posterior source files and a posterior sources configuration file

If you have posterior samples from a previous calibration and want to draw from those instead of uniform ranges, provide a `posterior_sources` YAML file and text files for the posterior sources (see below).

### 4. An ensemble configuration file

Either a Python dictionary (when using the library directly) or a YAML file (when using the CLI). See the examples below.

## Usage

### As a CLI

Run the ensemble generator from a YAML configuration file without writing any Python:

```bash
meristem paramgen config.yaml
```

where `config.yaml` contains the same keys as the `from_dict` dictionary below. For example:

```yaml
ensemble_type: LatinHypercube
param_dir: /path/to/param_dir
default_param_file: /path/to/fates_params_default.json
ensemble_dir: /path/to/output
file_prefix: my_ensemble
ensemble_members: 100
```

### As a Python library (e.g. in a Jupyter notebook)

#### Latin Hypercube ensemble

```python
from meristem.paramgen import ParamEnsemble
 
ensemble = ParamEnsemble.from_dict({
    "ensemble_type": "LatinHypercube",
    "param_dir": "/path/to/param_dir",
    "default_param_file": "/path/to/fates_params_default.json",
    "ensemble_dir": "/path/to/output",
    "file_prefix": "my_ensemble",
    "ensemble_members": 100,
})
 
ensemble.create_ensemble()
```

This produces 100 parameter files in `/path/to/output/`, named `my_ensemble_0000.json` through `my_ensemble_0099.json`, plus a key file `my_ensemble_key.csv` that records the normalized parameter value used in each member, and a text file `my_ensemble.txt` listing all member names.

#### One-at-a-time ensemble

```python
from meristem.paramgen import ParamEnsemble
 
ensemble = ParamEnsemble.from_dict({
    "ensemble_type": "OAT",
    "param_dir": "/path/to/param_dir",
    "default_param_file": "/path/to/clm_param_default.nc",
    "ensemble_dir": "/path/to/output",
    "file_prefix": "oat_run",
})
 
ensemble.create_ensemble()
```

This produces two files per parameter (one at minimum value, one at maximum), plus a key file with a `direction` column indicating `minimum` or `maximum` and a text file `oat_run.txt` listing
all member names.

#### Morris ensemble

```python
from meristem.paramgen import ParamEnsemble

ensemble = ParamEnsemble.from_dict({
    "ensemble_type": "Morris",
    "param_dir": "/path/to/param_dir",
    "default_param_file": "/path/to/fates_params_default.json",
    "ensemble_dir": "/path/to/output",
    "file_prefix": "morris_run",
    "morris_n": 20,
    "levels": 4,
})

ensemble.create_ensemble()
```

A Morris ensemble is a set of trajectories. Each trajectory starts at a random point on a
grid and then changes one parameter (or group) at a time, so with `D` parameters a
trajectory has `D + 1` members and the ensemble has `morris_n * (D + 1)` members in total.
Consecutive members within a trajectory are what Morris sensitivity analysis compares.

| Key | Default | Description |
| --- | --- | --- |
| `morris_n` | `100` | Number of trajectories to generate |
| `levels` | `4` | Number of grid levels each parameter's range is divided into. Must be a positive even number |
| `optimal_trajectories` | `None` | If set (between 2 and `morris_n`), select this many trajectories from the `morris_n` generated, chosen to maximize their spread. The ensemble then has `optimal_trajectories * (D + 1)` members |

Sampling uses [SALib](https://salib.readthedocs.io/). The key file records the sampled
normalized values in member order, so it can be passed to SALib's Morris analysis along
with your model output.

### Further use cases

#### Fixing indices at default

If you want to vary parameters across some indices but hold others at their default values, use `fixed_indices`:

```python
ensemble = ParamEnsemble.from_dict({
    "ensemble_type": "LatinHypercube",
    "param_dir": "/path/to/param_dir",
    "default_param_file": "/path/to/fates_params_default.json",
    "ensemble_dir": "/path/to/output",
    "file_prefix": "my_ensemble",
    "ensemble_members": 100,
    "fixed_indices": {"fates_pft": [7, 8, 9]},  # hold PFTs 8, 9, 10 at default (0-based)
})
```

#### Varying only a subset of parameters

If you have more parameters in your `main.csv` than you actually want to use in an
ensemble, you can restrict which ones are used with the `param_list` attribute:

```python
ensemble = ParamEnsemble.from_dict({
    "ensemble_type": "LatinHypercube",
    "param_dir": "/path/to/param_dir",
    "default_param_file": "/path/to/fates_params_default.json",
    "ensemble_dir": "/path/to/output",
    "file_prefix": "my_ensemble",
    "ensemble_members": 100,
    "param_list": ["fates_leaf_slatop", "fates_leaf_vcmax25top"],
})
```

This will produce an ensemble *only* using `fates_leaf_slatop` and `fates_leaf_vcmax25top`,
no matter how many parameters are listed in `main.csv`.

#### Using posterior distributions

If you have posterior samples from a previous calibration and want to draw from those instead of uniform ranges, provide a `posterior_sources` YAML file:

```python
ensemble = ParamEnsemble.from_dict({
    "ensemble_type": "LatinHypercube",
    "param_dir": "/path/to/param_dir",
    "default_param_file": "/path/to/fates_params_default.json",
    "ensemble_dir": "/path/to/output",
    "file_prefix": "my_ensemble",
    "ensemble_members": 100,
    "posterior_sources": "/path/to/posterior_sources.yaml",
})
```

#### Posterior sources YAML

The YAML file maps each parameter name (as it appears in `main.csv`) to a list of
posterior files and the PFT indices those files apply to:

```yaml
fates_leafn_vert_scaler:
  parameters:
    - fates_leafn_vert_scaler_coeff1
    - fates_leafn_vert_scaler_coeff2
  files:
    - path: /path/to/posterior_samples.txt
      array_indices: all
```

If `array_indices` is `'all'`, this means the posterior applies to every index along the parameter's
dimension (i.e. all PFTs get values drawn from the same distribution).

If `array_indices` is a 0-based integer, each file must contain columns for all parameters listed
under `parameters`. Together, the `array_indices` across all files should cover every
index you want sampled from a posterior — any index not covered will not be modified
from its default value

#### Per-index posteriors

If you have different posterior distributions for different PFTs (for example, because
you calibrated tropical and boreal PFTs separately) you can specify multiple files,
each covering a different set of indices:

```yaml
fates_leafn_vert_scaler:
  parameters:
    - fates_leafn_vert_scaler_coeff1
    - fates_leafn_vert_scaler_coeff2
  files:
    - path: /path/to/posterior_tropical.txt
      array_indices: [0, 1, 2, 3, 4]
    - path: /path/to/posterior_boreal.txt
      array_indices: [5, 6, 7, 8, 9]
```

#### Posterior file format

Each posterior file is a space-delimited text file with one column per parameter and
one row per posterior sample:

```text
fates_leafn_vert_scaler_coeff1 fates_leafn_vert_scaler_coeff2
0.012 2.1
0.015 2.5
0.008 1.9
```

## Output files

For each ensemble, paramgen writes:

* **`{prefix}_0000.nc/json`, `{prefix}_0001.nc/json`, ...**: one parameter file per ensemble member, each a modified copy of the default parameter file
* **`{prefix}_key.csv`**: a table recording the normalized value (0–1) each parameter took in each member; for OAT ensembles, also includes a `direction` column (`minimum` or `maximum`)
* **`{prefix}.txt`**: a plain text list of ensemble member names, one per line, for use in scripting

## Parameter types

paramgen supports four parameter types, set via the `param_type` column in `main.csv`:

**`default`**: the parameter is a standard NetCDF variable and is written directly. Works for both multi-dimensioned and scalar parameters.

**`sliced`**: the parameter targets one slice of a multi-dimensional variable. For example, varying only the `fates_leafage_class=0` slice of `fates_leaf_vcmax25top`.

**`scale_from_root`**: the parameter value is written as `root_param + delta`, where `delta` is what gets sampled. Used when two parameters must maintain a consistent offset relationship.

**`joint`**: multiple parameters are sampled together. Used for parameters that are correlated and should not be varied independently.

## Expand by index

Setting `expand_dim` on a parameter in `main.csv` tells paramgen to treat each index along that dimension as a separate, independently-sampled parameter. For example, setting `expand_dim = fates_pft` on `fates_leaf_slatop` will produce one parameter per PFT (`fates_leaf_slatop_0`, `fates_leaf_slatop_1`, etc.), each varying independently in the ensemble.

## Grouping parameters

Setting `group_name` on a parameter in `main.csv` links it to other parameters with the same group name. All parameters in a group receive the same normalized sample value in each ensemble member.

For a Latin Hypercube ensemble, each group occupies one column in the latin hypercube matrix rather than one column per parameter. The ensemble key records one value per group rather than per individual parameter.

For an OAT ensemble, all parameters in a group move to their minimum together, then their maximum together, as a single pair of ensemble members.

For a Morris ensemble, each group counts as one variable, so a trajectory has (number of groups and ungrouped parameters) + 1 members, and all parameters in a group step together.

**Constraints on groups:**

* All parameters in a group must have the same `expand_dim` (or all have none). Mixing expanded and unexpanded parameters in the same group is not allowed.
* A `scale_from_root` parameter and its root must be in the same group, or both ungrouped. Placing them in different groups would break the write-order dependency.
