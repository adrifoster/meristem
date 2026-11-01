# ensemble: submit and manage ensembles of CLM cases

`meristem.ensemble` manages ensembles of runs on a cluster.

After generating a suite of parameter files, `meristem.ensemble` can be used to generate
and manage an ensemble of CLM/CLM-FATES runs that use those parameter files.

----

## How it works

An ensemble is one `meristem.run` pipeline per ensemble member. Each member gets its own parameter
file from `meristem.paramgen`, and each of its stages (if it has more than one) runs as a CIME case
cloned from a single, already built base case. Within a member, multiple stages are tethered
exactly as in `meristem.run`, `mestistem.ensemble` adds the cloning and the parameter file setup around them.

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

`status` shows the state of every member and stage, `retry` resubmits failed or lost stages, and
`manifest` writes a CSV listing the members. See [Quick start](#quick-start) for how to use them.

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

### 1. Write a ensemble YAML

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

