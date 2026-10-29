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

### 1. Write a ensemble YAML