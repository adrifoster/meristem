#!/usr/bin/env bash
#
# Machine- and user-specific settings for the meristem.run smoke test.
#
# Copy this file to smoke_settings.sh (in the same directory) and edit the values.
# smoke_settings.sh is ignored by git, so your paths and project code stay local.
#
# It is sourced both by submit_smoke_test.sh and by the stage scripts inside the
# PBS jobs (which do not inherit your shell environment).

# path to your CTSM checkout
CTSM_ROOT=/glade/work/${USER}/CTSM

# CIME output root: case run directories and the short-term archive go here
OUT_DIR=/glade/derecho/scratch/${USER}/meristem_smoke

# PBS project code to charge
PROJECT=PXXXXXXXX

# NEON site to run
SITE=ABBY
