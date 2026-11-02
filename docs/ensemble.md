# ensemble: submit and manage ensembles of CLM cases

`meristem.ensemble` manages ensembles of runs on a cluster.

After generating a suite of parameter files, `meristem.ensemble` can be used to generate
and manage an ensemble of CLM/CLM-FATES runs that use those parameter files.

----

## How it works

An ensemble is one `meristem.run` pipeline per ensemble member. Each member gets its own parameter
file from `meristem.paramgen`, and each of its stages (if it has more than one) runs as a CIME case
cloned from a single, already built base case. Within a member, multiple stages are tethered
exactly as in `meristem.run`, `meristem.ensemble` adds the cloning and the parameter file setup around them.

### Creating and submitting

`meristem ensemble create` finds each member's parameter file and checks the base case and your stage
scripts (if you have any). It then writes the ensemble state file and a run state file for each member.
It doesn't submit any jobs. The ensemble can't be moved after this, because the state files and job scripts
contain absolute paths. If you need it somewhere else, create it again.

`meristem ensemble submit` submits the first stage's setup job for every member that hasn't started.
Use `--limit N` to submit only N members as a test. After that, each member's pipeline runs on its own
through PBS dependencies.

### What a member's stage does

Each stage of each member consists of the same jobs as in `meristem.run` (setup, CIME, advance and
fail). The setup job runs three things in order before the case is submitted:

1. **`meristem ensemble prepare`**: clones the base case with `create_clone --keepexe` and runs
   `case.setup`.
2. **Your stage script**, if the stage has one. It runs on the clone, so it can adjust the case for
   that stage. This would be the place you might put changes to the `user_nl_clm` file, or apply `xmlchange`s.
3. **`meristem ensemble finalize`**: sets the member's parameter file in the clone's `user_nl_clm`
   (using `paramfile_key`), then checks that the generated `lnd_in` points at that file.

Every stage of every member has its own case, named
`<base case>_<ensemble_id>_<member>_<stage>` and created under `case_dir`. Ensembles that
reuse the same base case and members don't collide, since `ensemble_id` is part of the name.

Because the ensemble supplies `prepare` and `finalize` itself, member stages can't set `command`,
`post_command`, `case_root` or `no_cime`.

### Monitoring and recovery

`status` shows the state of every member and stage, `retry` resubmits failed or lost stages,
`cancel` stops members' queued or running jobs (marking their stages failed, so `retry --failed`
resubmits them later), `manifest` writes a CSV listing the members, and `clean` deletes the
clones and other files of some or all members once you no longer need them. See
[Quick start](#quick-start) for how to use them.

### Where things live

- `<root>/ensemble_state.json`: the ensemble's state file.
- `<root>/members/<member>/`: each member's `meristem.run` root, with its `run_state.json` and the
  member's PBS job scripts and logs.
- `case_dir` (default `<root>/cases`): one clone per member stage, named
  `<base case>_<ensemble_id>_<member>_<stage>`.
- Each clone's CIME run directory and archive go wherever CIME puts them: under `output_root` if
  it's set (passed to `create_clone --cime-output-root`), otherwise CIME's default output root.

The base case and the paramgen directory are only read, never changed.

### Requirements

- A machine with PBS (e.g. Derecho) and CESM/CTSM set up on it
- A **built base case**: `BUILD_COMPLETE` must be `TRUE`. Every member is cloned from it with
  `--keepexe`, so nothing is rebuilt.
- **Paramgen output** for the ensemble in one directory (see [paramgen](paramgen.md)):
  - `<prefix>.txt`: the member names, one per line
  - `<prefix>_key.csv`: a table with an `ensemble` column naming each member
  - one parameter file per member, `<member>.nc` or `<member>.json` (exactly one of the two)
- meristem installed in a Python environment on a filesystem the compute nodes can see, as for
  [`meristem.run`](run.md#requirements). Don't move or rebuild that environment while the ensemble is
  running.

----

## Quick start

### 1. Write an ensemble YAML

```yaml
# ensemble.yaml
ensemble_id: morris_test                       # short identifier (letters, digits, hyphens, underscores)
root: /glade/derecho/scratch/user/morris_test  # ensemble state, job scripts and logs
project: PXXXXXXX                              # PBS project code
email: name@ucar.edu                           # email

base_case: /glade/work/user/cases/base_case    # built case every member is cloned from
paramfile_key: fates_paramfile                 # user_nl_clm variable: paramfile or fates_paramfile

paramgen:
  dir: /glade/work/user/morris_files           # paramgen output directory
  prefix: morris                               # reads morris.txt and morris_key.csv
  members: all                                 # or a list, e.g. ["0001", "0002"]

member_stages:                                 # run once per member, in order
  - name: spinup
    queue: develop
  - name: transient
    script: /path/to/setup_transient.sh        # optional
    queue: develop
```

Quote member names that are all digits (`"0001"`) so YAML doesn't read them as numbers.

A stage's `script` is optional. If given, it runs on the member's clone and is called as
`bash <script> <case_root> <previous_case_name> <previous_case_root>`.

### 2. Initialise the ensemble

```bash
meristem ensemble create --config ensemble.yaml
```

This checks the base case, parameter files and stage scripts, then writes the state files. No jobs
are submitted.

### 3. Submit

```bash
meristem ensemble submit --root /glade/derecho/scratch/user/morris_test
```

This submits the first stage of every member. Add `--limit 2` to submit only two members as a test
(running the same command again later submits the rest), or `--dry-run` to write the job scripts
without calling `qsub`.

### 4. Check status

```bash
meristem ensemble status --root /glade/derecho/scratch/user/morris_test
```

```text
Ensemble: morris_test (/glade/derecho/scratch/user/morris_test)
50 members

           pending  submitted  done  failed
spinup           0         10    38       2
transient       12          0     0       0

Failed (2): 0007:spinup, 0031:spinup
  retry with: meristem ensemble retry --root /glade/derecho/scratch/user/morris_test --failed
```

Add `--print-members` to list every member and stage, or `--members 0001 0002` to check particular
members. Add `--check-queue` to ask PBS about each submitted stage and report any that are **lost**
(see [Lost stages](run.md#lost-stages)). `status` exits with code 3 if any stage is lost.

### 5. Retry

```bash
meristem ensemble retry --root /glade/derecho/scratch/user/morris_test --failed
```

`--failed` and `--lost` retry every stage in that state. To resubmit specific stages, name them with
`--stage <name>` and/or `--members 0001 0002`. Add `--skip-script` to resubmit without re-running the
stage script.

### 6. Write a manifest

```bash
meristem ensemble manifest --root /glade/derecho/scratch/user/morris_test
```

This writes `<root>/manifest.csv` (or the file given with `-o`), with one row per member: the
paramgen key values, its parameter file, and each stage's status, job ID, attempts, case root and
times.

### 7. Cancel

```bash
meristem ensemble cancel --root /glade/derecho/scratch/user/morris_test
```

This cancels the queued or running jobs (setup, CIME, advance and fail) of every member, or only
the members named with `--members 0001 0002`, and marks the stages that were running as failed.
Use `--dry-run` to see what would be cancelled without actually cancelling anything. To run the
members again later, use `meristem ensemble retry --failed`.

### 8. Clean

```bash
meristem ensemble clean --root /glade/derecho/scratch/user/morris_test
```

With no options, this deletes the whole ensemble: every member's clones and their CIME run
directories (RUNDIR), then the ensemble root itself, including the state files, job scripts, logs
and any `manifest.csv` there (write the manifest somewhere else first with `-o` if you want to keep
it). The base case, the paramgen directory and a `case_dir` outside the root are left in place;
only the clones inside `case_dir` are deleted.

To clean part of the ensemble, name members with `--members 0001 0002` and/or stages with
`--stages spinup transient`. Only those member stages' clones, run directories and job scripts/logs
are deleted. The members' state files are kept, so `status`, `manifest` and `retry` still work, and
the stages' status is left as it is (use `retry` to rerun one).

Add `--include-archive` to also delete the clones' archive directories (`DOUT_S_ROOT`). For safety,
a RUNDIR or archive whose path doesn't contain the clone's case name is never deleted, and the
build directory (`EXEROOT`, shared with the base case through `--keepexe`) is never touched.

`clean` shows how many paths it will delete, with a few examples, and asks for confirmation;
`--yes` skips the question. `--dry-run` lists every path and deletes nothing. If any selected
member has jobs queued or running, `clean` refuses and deletes nothing in any member; add
`--cancel` to cancel those jobs first, which marks their stages failed just as `cancel` does.

----

## Config reference

### Top-level keys

| Key | Required | Default | Description |
| --- | --- | --- | --- |
| `ensemble_id` | yes | | Short identifier (letters, digits, hyphens, underscores), used in case and job names |
| `root` | yes | | Directory for the ensemble's and members' state files, job scripts and logs |
| `base_case` | yes | | Built CIME case (`BUILD_COMPLETE` is `TRUE`) that every member is cloned from |
| `paramfile_key` | yes | | `user_nl_clm` variable to set to each member's parameter file: `paramfile` or `fates_paramfile` |
| `paramgen` | yes | | Where to find the parameter files (see below) |
| `member_stages` | yes | | Stages run once per member, in order (see below) |
| `case_dir` | no | `<root>/cases` | Directory where member clones are created |
| `output_root` | no | CIME's default | CIME output root for the clones (`create_clone --cime-output-root`) |
| `project` | no | `$PROJECT` | PBS project/account code. If omitted, the `PROJECT` environment variable is used |
| `email` | no | | User email for job emails. If omitted, no email directive is added |
| `helper_queue` | no | each stage's `queue` | PBS queue for the members' small advance and fail jobs, e.g. `develop` |

Paths must be absolute. Environment variables in the YAML file (`$VAR` or `${VAR}`) are expanded
when the ensemble is created.

### `paramgen` keys

| Key | Required | Default | Description |
| --- | --- | --- | --- |
| `dir` | yes | | `meristem paramgen` output directory |
| `prefix` | yes | | Paramgen prefix: reads `<prefix>.txt` and `<prefix>_key.csv` |
| `members` | no | `all` | `all`, or a list of member names to run (quote all-digit names, e.g. `"0001"`) |

### Member stage keys

The same as `meristem.run` [stage keys](run.md#stage-keys), except that `script` is optional and
`command`, `post_command`, `case_root` and `no_cime` aren't allowed: the ensemble sets the case
root and runs `prepare` and `finalize` around the script itself.

----

## Command line reference

```text
meristem ensemble create --config ensemble.yaml
  Check the base case, parameter files and stage scripts, and write the state files.
  Submits nothing.

meristem ensemble submit --root <path> [--limit N] [--dry-run]
  Submit the first stage of every member that hasn't started (at most N with --limit).

meristem ensemble status --root <path> [--members <m1> <m2> ...] [--print-members] [--check-queue]
  Print a count of member stages in each state, and the failed (and lost) ones.
  --print-members also lists every member and stage; --check-queue asks PBS about
  submitted stages and reports lost ones. Exits 3 if a stage is lost, 0 otherwise.

meristem ensemble retry --root <path> [--failed] [--lost] [--stage <name>] [--members <m1> <m2> ...] [--skip-script] [--dry-run]
  Resubmit failed (--failed) and/or lost (--lost) stages, or the stages named with
  --stage and/or --members. --skip-script resubmits without re-running the stage script.

meristem ensemble cancel --root <path> [--members <m1> <m2> ...] [--dry-run]
  Cancel the queued or running jobs of every member (or those named) and mark their
  submitted stages as failed; --dry-run just reports what would be cancelled.

meristem ensemble clean --root <path> [--members <m1> <m2> ...] [--stage/--stages <name1> <name2> ...]
        [--include-archive] [--cancel] [--dry-run] [--yes]
  Delete an ensemble's files. With neither --members nor --stages, deletes every
  member's clones and CIME run directories (RUNDIR), then the whole ensemble root,
  including the state files and any manifest.csv there.
  With --members and/or --stages, deletes only those member stages' clones, run
  directories and job scripts/logs; the members' state files are kept, and the
  stages' status is left as it is (use retry to rerun one), unless --cancel
  cancels their jobs.
  --include-archive also deletes the clones' archive directories (DOUT_S_ROOT).
  The base case, the paramgen directory and case_dir itself are never deleted.
  For safety, a RUNDIR or archive whose path doesn't contain the case name is
  never deleted, and the build directory (EXEROOT) is never touched.
  Shows a summary of what it will delete and asks for confirmation; --yes skips
  the question, and --dry-run lists every path without deleting anything.
  Refuses, deleting nothing, if any selected member's jobs are queued or running;
  --cancel first cancels them and marks their submitted stages as failed,
  exactly as `meristem ensemble cancel` does.

meristem ensemble manifest --root <path> [-o <file>]
  Write a CSV with one row per member (default <root>/manifest.csv).

meristem ensemble prepare --root <path> --member <member> --stage <name>
  Clone the base case for one member stage and run case.setup. Called automatically
  from inside the setup job; not typically run manually.

meristem ensemble finalize --root <path> --member <member> --stage <name>
  Set the member's parameter file in its clone and check it. Called automatically
  from inside the setup job; not typically run manually.
```

Add `--dry-run` to `submit` or `retry` to write the job scripts without calling `qsub`, to
`cancel` to see what would be cancelled, or to `clean` to see what would be deleted. A dry run
doesn't change the ensemble's state.

Add `--debug` to any command to show the full traceback on errors.
