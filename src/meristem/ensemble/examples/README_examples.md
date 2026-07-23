# Parameter Ensemble Examples

These examples demonstrate how to use `param_ens_gen` to generate parameter ensembles
for either CLM (`examples/clm_example`) and FATES (`examples/fates_example`)

It includes two examples: a Latin Hypercube ensemble and a One-at-a-time ensemble.

## Contents

### CLM Example (`clm_example/`)

```text
clm_example/
    clm_param_dir/       parameter metadata (main.csv and per-PFT bound files)
    default_clm_param.nc default CLM parameter file
    lh_config.yaml       Latin Hypercube ensemble configuration
    oat_config.yaml      One-at-a-time ensemble configuration
```

### FATES Example (`fates_example/`)

```text
fates_example/
    fates_param_dir/          parameter metadata (main.csv and per-PFT bound files)
    default_fates_param.json  default FATES parameter file
    posterior_sources.yaml    example posterior sources configuration
    leafn_vert_scaler.txt     example posterior sources data file
    lh_config.yaml            Latin Hypercube ensemble configuration
    oat_config.yaml           One-at-a-time ensemble configuration
```


## Setup

From the repo root, install the package and activate your environment:

```bash
conda activate param_ens_gen
```

## Running the examples

All commands should be run from inside the example directory, e.g.::

```bash
cd examples/clm_example
```

### Latin Hypercube ensemble

Generates 20 ensemble members by sampling all parameters simultaneously using a Latin Hypercube design.

```bash
param_ens_gen run lh_config.yaml
```

Output it written to `output_LH/`. This produces:

* `clm_lh_000.nc/json` through `clm_lh_019.nc/json`: one parameter file per ensemble member
* `clm_lh_key.csv`: normalized parameter values for each member
* `clm_lh.txt`: list of ensemble member names, for use in scripting

### One-at-a-time ensemble

Generates two ensemble members per parameter: one at its minimum value and one at its maximum value. All other parameters are held at default.

```bash
param_ens_gen run oat_config.yaml
```

Output is written to `output_OAT/`. This produces:

* `clm_lh_000.nc/json` through `clm_lh_00n.nc/json`: One parameter file per parameter per direction
* `clm_oat_key.csv`: parameter name and direction for each member
* `clm_oat.txt`: list of ensemble member names

## Output format

The output parameter files are written in the same format as the input — NetCDF (`.nc`) for CLM, JSON (`.json`) for FATES.

## Parameters

All parameters defined in in main.csv's are varied in all ensembles. See the main `param_ens_gen` README for a full description of the parameter metadata format and supported parameter types.
