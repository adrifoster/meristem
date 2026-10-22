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
- meristem installed in a Python environment on a filesystem the compute nodes can see (e.g.
  `/glade/work`). Generated job scripts call meristem through the absolute path of the Python
  that created the run (`/path/to/python -m meristem`), so no environment needs to be activated
  inside the jobs. Don't move, delete, or rebuild that environment while a run is in progress.
- Setup scripts can call `meristem` directly. They do not get an activated conda environment,
  so a setup script that needs one must activate it itself.

----

## Quick start

### 1. Write a config YAML

```yaml
# run.yaml
root: /glade/derecho/scratch/user/my_spinup  # working directory for the entire run
run_id: my_spinup                            # short identifier (letters, digits, hyphens, underscores)
project: PXXXXXXX                            # PBS project code
email: name@ucar.edu                         # email

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

`--print-status` shows what the state file records. If a stage has been `submitted` for longer than
you'd expect, use `--check` to ask PBS whether it is still running (see [Lost stages](#lost-stages)).

----

## Config reference

### Top-level keys

| Key | Required | Default | Description |
| --- | --- | --- | --- |
| `root` | yes | | Working directory for the run. Job scripts, logs, the state file and each stage's case live here. Must not contain whitespace, since PBS writes job logs here |
| `stages` | yes | | List of stages, run in order (see below) |
| `run_id` | no | `run` | Short identifier used in job names |
| `project` | no | `$PROJECT` | PBS project/account code (`#PBS -A`). If omitted, the `PROJECT` environment variable is used, and it's an error if neither is set |
| `email` | no | user email, used for job emails (`#PBS -M`). If omitted, the directive is not added |
| `scheduler` | no | `pbs` | Job scheduler used to submit job scripts. Saved in the state file. Only `pbs` is currently supported |

Environment variables in the YAML file (`$VAR` or `${VAR}`) are expanded when the run is created,
e.g. `root: /glade/derecho/scratch/$USER/my_spinup`.

### Stage keys

| Key | Required | Default | Description |
| --- | --- | --- | --- |
| `name` | yes | | Unique stage name. By default the stage's case is created at `<root>/<name>` |
| `script` | yes | | Path to the setup script for this stage (see [Setup scripts](#setup-scripts)) |
| `queue` | yes | | PBS queue for the setup job |
| `case_root` | no | `<root>/<name>` | Absolute path to the stage's case directory |
| `walltime` | no | `01:00:00` | Walltime for the setup job (`HH:MM:SS`) |
| `select` | no | `1` | Number of nodes for the setup job |
| `ncpus` | no | `1` | CPUs per node for the setup job |
| `memory` | no | `10GB` | Memory for the setup job, e.g. `16GB` |
| `no_cime` | no | `false` | If true, the stage isn't a CIME case: run the script, then advance directly |
| `extra_pbs` | no | `[]` | Extra raw PBS directives, e.g. `["-l place=group=rack"]` |
| `spinup_check` | no | `false` | Reserved for future use. Currently has no effect |

The resource keys (`walltime`, `select`, `ncpus`, `memory`) apply to the setup job only. The model run's
own resources come from the CIME case, as usual.

----

## Command line reference

```text
meristem run --create --config run.yaml
  Initialise a new run from a YAML config file.

meristem run --root <path> --print-status
  Print the current status of all stages, as recorded in the state file.
  Doesn't contact PBS.

meristem run --root <path> --check
  Ask PBS about each submitted stage and print the status, showing any
  lost stage as "lost" (see Lost stages below). Changes nothing. Exits 3
  if a stage is lost, 0 otherwise.

meristem run --root <path> --submit [--dry-run]
  Submit the first pending stage.

meristem run --root <path> --retry [--stage <name>] [--dry-run] [--skip-script] [--lost]
  Reset and resubmit a pending or failed stage. If --stage is omitted,
  retries the current (first non-complete) stage. If --skip-script, it
  resubmits without re-running the setup script. With --lost, it instead
  retries a submitted stage that is lost, and refuses if it isn't.

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
A dry run doesn't change the run's state, so you can follow it with the real command.
Add `--debug` to any command to show the full traceback on errors.

### Lost stages

A stage is **lost** when the state file says `submitted` but no job in PBS will ever change
that. Normally the stage's advance job marks it done, or its fail job marks it failed. If
neither runs, the stage stays `submitted` for good. This happens when:

- the setup job is killed (for example at walltime) before it submits the advance and fail jobs.
  The setup job's error trap doesn't fire when PBS kills it.
- the advance or fail job is deleted with `qdel`, or killed, before it updates the stage.

meristem records the setup, advance and fail job IDs for each stage. A stage is lost if it is
`submitted` and **none** of those jobs is still queued or running. While the model is running, the
advance and fail jobs sit in the queue waiting on it, so a stage whose model run is still going is
never lost.

To find lost stages:

```bash
meristem run --root <path> --check
```

```text
Run: my_spinup (/glade/derecho/scratch/user/my_spinup)
  ✓ spinup_ad             done      [98765.desched1]
  ? spinup_postad         lost      [98770.desched1]
Stage 'spinup_postad' is lost. Retry with: meristem run --root <path> --retry --stage spinup_postad --lost
```

`--check` exits with code 3 when it finds a lost stage, so a script can loop over many runs and
pick out the ones that need attention. If PBS can't be queried, it exits with code 1 and changes
nothing.

To retry a lost stage:

```bash
meristem run --root <path> --retry --stage <name> --lost
```

This checks PBS again first and refuses if any of the stage's jobs is still queued or running, so it
can't start a second copy of a stage that is still going.

**Known gaps.** In two cases a stage can be reported lost while its model run is still going.
Check with `qstat -u $USER` before retrying:

- The setup job was killed in the few seconds between submitting the case and submitting the
  advance and fail jobs.
- The run was created with a meristem version from before advance and fail job IDs were recorded,
  so only the setup job is known.

**Retrying by hand.** If you're sure nothing is still running for a stage, you can also mark it
failed yourself and retry it, without asking PBS:

```bash
meristem run --root <path> --fail --stage <name>
meristem run --root <path> --retry --stage <name>
```

----

## Setup scripts

Each stage requires a setup script that creates, configures and builds the case. The setup job calls it as:

```bash
bash <script> <case_root> <previous_case_name> <previous_case_root>
```

- `case_root` is where the script should create the case: the stage's `case_root` if set, otherwise
  `<root>/<stage name>`.
- `previous_case_name` is the CIME case name of the previous stage: the last component of its
  `case_root`, which is the previous stage's name unless it set a custom `case_root`. Use it to start
  from the previous stage's restart files, e.g. `./xmlchange RUN_REFCASE=$2`.
- `previous_case_root` is the full path to the previous stage's case directory.

For the first stage, neither previous-case argument is passed, so `$#` is 1.

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

