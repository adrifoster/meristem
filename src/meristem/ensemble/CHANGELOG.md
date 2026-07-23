# Changelog

All notable changes to this project will be documented here.

## [0.1.0] - 2026-06-09

### Added

* Latin Hypercube and One-at-a-time ensemble generation
* NetCDF and FATES JSON parameter file support
* Parameter types: `default`, `sliced`, `scale_from_root`, `joint`
* Parameter grouping via `group_name`: grouped parameters share the same normalized sample value
* Per-index expansion via `expand_dim`: independently vary each PFT or other dimension index
* Fixed indices: hold specific dimension indices at their default values
* Posterior distribution sampling from external files
* PFT-specific parameter bounds via per-parameter CSV sheets
* Command-line interface: `param_ens_gen run config.yaml`
* Diagnostic tools: `normalize_defaults` and `plot_param_bounds`
* Examples for CLM (NetCDF) and FATES (JSON) parameter files

