#!/usr/bin/env bash
#
# Setup script for the accelerated-decomposition (AD) spinup stage.
# Called by meristem.jobs as: bash run_job_ad.sh <case_root> <previous_stage_name>
# (previous_stage_name is empty for this first stage)

# machine- and user-specific settings: CTSM_ROOT, OUT_DIR, PROJECT, SITE
source "$(dirname "${BASH_SOURCE[0]}")/../smoke_settings.sh"

COMPSET=2000_DATM%1PT_CLM60%BGC_SICE_SOCN_SROF_SGLC_SWAV
RES=CLM_USRDAT

if [ $# -lt 1 ]; then
  echo "ERROR: please specify a case_root"
  exit 1
fi
case_root="$1"
case_name=$(basename "${case_root}")

# start clean if this stage is being retried
rm -rf "${case_root}" "${OUT_DIR:?}/${case_name}" "${OUT_DIR:?}/archive/${case_name}"

user_mods=${CTSM_ROOT}/cime_config/usermods_dirs/clm/NEON/${SITE}

cd "${CTSM_ROOT}/cime/scripts"
./create_newcase --case "${case_root}" --compset "${COMPSET}" --res "${RES}" \
  --project "${PROJECT}" --run-unsupported --output-root "${OUT_DIR}" \
  --user-mods-dir "${user_mods}"
cd "${case_root}"

./xmlchange --subgroup case.run JOB_WALLCLOCK_TIME=01:00:00
./xmlchange --subgroup case.st_archive JOB_WALLCLOCK_TIME=01:00:00
./xmlchange STOP_OPTION="ndays"
./xmlchange STOP_N=10
./xmlchange RESUBMIT=0

./xmlchange CLM_ACCELERATED_SPINUP="on"
./xmlchange CLM_FORCE_COLDSTART=on

./case.setup
./preview_namelists
./case.build

### DO NOT DO ./case.submit: meristem.jobs submits the case ###
