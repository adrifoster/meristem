# run: tethered multi-stage runs on PBS

`meristem.run` manages multi-stage CLM runs on PBS clusters.

A CLM run often needs several stages in order (for example accelerated-decomposition spinup, then
post-AD spinup, then a transient run), and each stage is normally set up and submitted by hand once the
previous one finishes. `meristem.run` *tethers* these stages together: each stage automatically
submits the next when it completes, using PBS job dependencies to ensure correct ordering.

> **Status**: Early development. The API may change.

----

## How it works

Each stage consists of up to four PBS jobs:

1. **Setup job**: runs your setup script for the stage (create, configure and build the case),
   submits the CIME case, then submits the advance and fail jobs below.
2. **CIME job**: the actual model run, submitted with `./case.submit --resubmit-immediate`.
3. **Advance job**: a lightweight bookkeeping job that runs `afterok` the CIME job. It marks the
   stage complete and submits the next stage's setup job.
4. **Fail job**: a lightweight bookkeeping job that runs `afternotok` the CIME job. If the model run
   dies, it marks the stage as failed so the pipeline stops cleanly.

If the setup script itself fails, a trap in the setup job marks the stage as failed.

Stages with `no_cime: true` skip steps 2–4. The setup job runs your script and then advances to the
next stage directly. This is useful for checks or post-processing steps that aren't CIME cases.

Only the first stage needs to be submitted manually. The rest of the pipeline runs automatically.

### Requirements

- A machine with PBS (e.g. Derecho)
- CESM/CTSM set up on that machine
- meristem installed in a conda environment named `meristem`. The generated job scripts run
  `conda activate meristem` so that `meristem run` is available inside each job.

----

## Quick start

### 1. Write a config YAML

```yaml
# run.yaml
root: /glade/derecho/scratch/user/my_spinup  # working directory for the entire run
run_id: my_spinup                            # short identifier (letters, digits, hyphens, underscores)
project: PXXXXXXX                            # PBS project code
user: auser                                  # PBS username (for job emails)

stages:
  - name: spinup_ad               # name of stage - used for job and case names
    script: /path/to/setup_ad.sh  # path to your setup script for this stage
    queue: develop                # PBS queue

  - name: spinup_postad
    script: /path/to/setup_postad.sh
    queue: develop
```

### 2. Initialise the run

```bash
meristem run --create --config run.yaml
```

This creates the run directory and writes the initial state file (`run_state.json`).

### 3. Submit the pipeline

```bash
meristem run --root /glade/derecho/scratch/user/my_spinup --submit
```

This submits the first stage. The rest of the pipeline runs automatically via PBS dependencies.

### 4. Check status

```bash
meristem run --root /glade/derecho/scratch/user/my_spinup --print-status
```

Output:

```text
Run: my_spinup (/glade/derecho/scratch/user/my_spinup)
  ✓ spinup_ad             done
  ... spinup_postad       submitted   [98765.pbs]
```

----

## Config reference

### Top-level keys

| Key | Required | Default | Description |
| --- | --- | --- | --- |
| `root` | yes | | Working directory for the run. Job scripts, logs, the state file and each stage's case live here |
| `stages` | yes | | List of stages, run in order (see below) |
| `run_id` | no | `run` | Short identifier used in job names |
| `project` | no | `$PROJECT` | PBS project/account code (`#PBS -A`). If omitted, the `PROJECT` environment variable is used, and it's an error if neither is set |
| `user` | no | `$USER` | PBS username, used for job emails (`#PBS -M`). If omitted, the `USER` environment variable is used |

Environment variables in the YAML file (`$VAR` or `${VAR}`) are expanded when the run is created,
e.g. `root: /glade/derecho/scratch/$USER/my_spinup`.

### Stage keys

| Key | Required | Default | Description |
| --- | --- | --- | --- |
| `name` | yes | | Unique stage name. The stage's case is created at `<root>/<name>` |
| `script` | yes | | Path to the setup script for this stage (see [Setup scripts](#setup-scripts)) |
| `queue` | yes | | PBS queue for the setup job |
| `walltime` | no | `01:00:00` | Walltime for the setup job (`HH:MM:SS`) |
| `select` | no | `1` | Number of nodes for the setup job |
| `ncpus` | no | `1` | CPUs per node for the setup job |
| `memory` | no | `10GB` | Memory for the setup job, e.g. `16GB` |
| `no_cime` | no | `false` | If true, the stage isn't a CIME case: run the script, then advance directly |
| `extra_pbs` | no | `[]` | Extra raw PBS directives, e.g. `["-l place=group=rack"]` |
| `kind` | no | `custom` | Reserved for future use (see [Stage kinds](#stage-kinds)) |
| `spinup_check` | no | `false` | Reserved for future use. Currently has no effect |

The resource keys (`walltime`, `select`, `ncpus`, `memory`) apply to the setup job only. The model run's
own resources come from the CIME case, as usual.

----

## Command line reference

```text
meristem run --create --config run.yaml
  Initialise a new run from a YAML config file.

meristem run --root <path> --print-status
  Print the current status of all stages.

meristem run --root <path> --submit [--dry-run]
  Submit the first pending stage.

meristem run --root <path> --retry [--stage <name>] [--dry-run] [--skip-script]
  Reset and resubmit a pending or failed stage. If --stage is omitted,
  retries the current (first non-complete) stage. If --skip-script, it
  resubmits without re-running the setup script.

meristem run --root <path> --submit-advance --stage <name> --cime-job-id <id>
  Submit the advance job for a stage. Called automatically
  from inside the setup job; not typically run manually.

meristem run --root <path> --submit-fail --stage <name> --cime-job-id <id>
  Submit the fail job for a stage, in case the CIME job dies. Called
  automatically from inside the setup job; not typically run manually.

meristem run --root <path> --advance --stage <name>
  Mark a stage complete and submit the next stage. Called
  automatically from inside the advance job.

meristem run --root <path> --fail --stage <name>
  Mark a stage as failed. Called automatically from the fail job
  and the setup job's error trap, or can be run manually.
```

Add `--dry-run` to any submission command to generate job scripts without calling `qsub`.
Add `--debug` to any command to show the full traceback on errors.

**Retrying a stuck stage.** `--retry` only accepts stages that are pending or failed. If a stage is
stuck as `submitted` (for example, you deleted its jobs with `qdel`), mark it failed first:

```bash
meristem run --root <path> --fail --stage <name>
meristem run --root <path> --retry --stage <name>
```

----

## Setup scripts

Each stage requires a setup script that creates, configures and builds the case. The setup job calls it as:

```bash
bash <script> <case_root> <previous_stage_name>
```

- `case_root` is where the script should create the case: `<root>/<stage name>`.
- `previous_stage_name` is the name of the previous stage, which is also its case name. Use it to
  start from the previous stage's restart files, e.g. `./xmlchange RUN_REFCASE=$2`. For the first
  stage it isn't passed at all, so `$#` is 1. The previous stage's case is at `$(dirname "$1")/$2`.

The script runs with `set -euo pipefail` enforced, so any failing command fails the stage.

> [!WARNING]
> Do NOT add `./case.submit` to the end of your scripts! The pipeline takes care of this.

## Example pipeline (smoke test)

`tests/run/smoke` contains a complete example pipeline, which also serves as an end-to-end test on a
PBS machine: a NEON single-site (ABBY) AD → SASU → post-AD spinup, followed by a `no_cime` check stage.

- `smoke_test.yaml`: the run config.
- `scripts/`: `run_job_ad.sh`, `run_job_sasu.sh` and `run_job_postad.sh` are example setup scripts
  you can adapt. The SASU and post-AD scripts show how to start from the previous stage's restart.
  `smoke_check.sh` is an example `no_cime` stage.
- `smoke_settings.example.sh`: the machine- and user-specific settings (CTSM checkout, output
  directory, project code) that the scripts read.

To run it:

```bash
cp tests/run/smoke/smoke_settings.example.sh tests/run/smoke/smoke_settings.sh
# edit smoke_settings.sh for your machine
bash tests/run/smoke/submit_smoke_test.sh
```

When all stages are done, `python tests/run/smoke/check_smoke_test.py <root>` confirms that the
pipeline completed. The submit script prints the exact command.

Note that PBS jobs don't inherit your shell's environment. That's why the example scripts read their
settings from a file next to them rather than from exported variables.

----

## Stage kinds

The `kind` key labels a stage. It doesn't change any behavior yet, but is reserved for stage-specific
behavior (such as spinup checks) in the future.

| Kind | Description |
| --- | --- |
| `ad` | Accelerated decomposition spinup |
| `sasu` | SASU spinup |
| `post-sasu` | Post-SASU stage |
| `historical` | Historical run |
| `custom` | Any other stage type (default) |
