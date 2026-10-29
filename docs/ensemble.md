# ensemble: submit and manage ensembles of CLM cases

`meristem.ensemble` manages ensembles of runs on clusters.

A CLM run often needs several stages in order (for example accelerated-decomposition spinup, then
post-AD spinup, then a transient run), and each stage is normally set up and submitted by hand once the
previous one finishes. `meristem.run` *tethers* these stages together: each stage automatically
submits the next when it completes, using job scheduler dependencies to ensure correct ordering.

----

## How it works

