# paramgen examples

These examples show how to use `meristem.paramgen` to generate parameter ensembles for CLM
(`clm_example/`) and CLM-FATES (`fates_example/`).

Each example has three configurations, one per sampling design: Latin Hypercube, one-at-a-time (OAT)
and Morris. The FATES example also shows how to draw a parameter from posterior samples.

## Contents

### CLM example (`clm_example/`)

```text
clm_example/
    clm_param_dir/        parameter metadata (main.csv and per-PFT bound files)
    default_clm_param.nc  default CLM parameter file
    lh_config.yaml        Latin Hypercube ensemble configuration
    oat_config.yaml       One-at-a-time ensemble configuration
    morris_config.yaml    Morris ensemble configuration
```

69 parameters, some of them grouped (e.g. the three acclimation scale factors share the group
`ACCLIM_SF`), giving 55 independently sampled groups and ungrouped parameters.

### FATES example (`fates_example/`)

```text
fates_example/
    fates_param_dir/          parameter metadata (main.csv and per-PFT bound files)
    default_fates_param.json  default FATES parameter file
    posterior_sources.yaml    posterior sources configuration
    leafn_vert_scaler.txt     posterior samples for the fates_leafn_vert_scaler coefficients
    lh_config.yaml            Latin Hypercube ensemble configuration
    oat_config.yaml           One-at-a-time ensemble configuration
    morris_config.yaml        Morris ensemble configuration
```

54 parameters. `fates_leafn_vert_scaler` is a `joint` parameter whose two coefficients are drawn
together from the posterior samples in `leafn_vert_scaler.txt`, rather than from a uniform range.

## Setup

Install meristem and activate its environment (see the main [README](../../README.md#installation)):

```bash
conda activate meristem
```

## Running the examples

Run all commands from inside an example directory, because the configs use paths relative to it:

```bash
cd examples/paramgen/clm_example     # or examples/paramgen/fates_example
```

The commands and output names below are for the CLM example. For FATES, replace `clm_` with `fates_`
and `.nc` with `.json`.

### Latin Hypercube ensemble

Generates 20 ensemble members by sampling all parameters simultaneously using a Latin Hypercube design.

```bash
meristem-paramgen run lh_config.yaml
```

Output is written to `output_LH/`:

* `clm_lh_0000.nc` through `clm_lh_0019.nc`: one parameter file per ensemble member
* `clm_lh_key.csv`: the normalized value (0–1) each parameter or group took in each member
* `clm_lh.txt`: list of ensemble member names, for use in scripting

### One-at-a-time ensemble

Generates two ensemble members per parameter (or group): one at its minimum value and one at its
maximum value, with all other parameters held at default.

```bash
meristem-paramgen run oat_config.yaml
```

Output is written to `output_OAT/`:

* `clm_oat_0000.nc`, `clm_oat_0001.nc`, ...: two parameter files per parameter or group (110 for CLM, 108 for FATES)
* `clm_oat_key.csv`: the parameter and direction (`minimum` or `maximum`) for each member
* `clm_oat.txt`: list of ensemble member names

### Morris ensemble

Generates `morris_n = 4` trajectories for Morris (elementary effects) sensitivity analysis. Each
trajectory changes one parameter or group at a time, so it has one member per parameter or group,
plus a starting point.

```bash
meristem-paramgen run morris_config.yaml
```

Output is written to `output_Morris/`:

* `clm_morris_0000.nc`, `clm_morris_0001.nc`, ...: one parameter file per member, `4 × (55 + 1) = 224` for CLM (`4 × (54 + 1) = 220` for FATES)
* `clm_morris_key.csv`: the normalized value each parameter or group took in each member, in member order
* `clm_morris.txt`: list of ensemble member names

A real Morris analysis usually uses more trajectories (`morris_n` of 10–50 is common). The example keeps it
small so it runs quickly.

## Output format

The output parameter files are written in the same format as the default parameter file: NetCDF (`.nc`)
for CLM, JSON (`.json`) for FATES. The output folders are ignored by git, so you can rerun the examples freely.

## Parameters

All parameters listed in each example's `main.csv` are varied in every ensemble. See
[docs/paramgen.md](../../docs/paramgen.md) for a full description of the `main.csv` format, parameter
types, grouping and posterior sources.
