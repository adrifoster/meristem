#!/usr/bin/env bash
#
# Setup script for the SASU spinup stage, starting from the previous stage's restart.
# Called by meristem.run as: bash run_job_sasu.sh <case_root> <previous_stage_name>

# machine- and user-specific settings: CTSM_ROOT, OUT_DIR, PROJECT, SITE
source "$(dirname "${BASH_SOURCE[0]}")/../smoke_settings.sh"

COMPSET=2000_DATM%1PT_CLM60%BGC_SICE_SOCN_SROF_SGLC_SWAV
RES=CLM_USRDAT

if [ $# -lt 2 ] || [ -z "$2" ]; then
  echo "ERROR: please specify case_root and previous_stage_name"
  exit 1
fi
case_root="$1"
prev_case="$2"
case_name=$(basename "${case_root}")

# start clean if this stage is being retried
rm -rf "${case_root}" "${OUT_DIR:?}/${case_name}" "${OUT_DIR:?}/archive/${case_name}"

user_mods=${CTSM_ROOT}/cime_config/usermods_dirs/clm/NEON/${SITE}

cd "${CTSM_ROOT}/cime/scripts"
./create_newcase --case "${case_root}" --compset "${COMPSET}" --res "${RES}" \
  --project "${PROJECT}" --run-unsupported --output-root "${OUT_DIR}" \
  --user-mods-dir "${user_mods}"
cd "${case_root}"

# start from the last restart of the previous stage
ref_rest_dir=${OUT_DIR}/archive/${prev_case}/rest
last_date=$(ls "${ref_rest_dir}" | tail -n1)
ref_dir=${ref_rest_dir}/${last_date}
ref_date=${last_date%-*}

./xmlchange RUN_TYPE=hybrid
./xmlchange RUN_REFCASE="${prev_case}"
./xmlchange GET_REFCASE="True"
./xmlchange RUN_REFDIR="${ref_dir}"
./xmlchange RUN_REFDATE="${ref_date}"

./xmlchange --subgroup case.run JOB_WALLCLOCK_TIME=01:00:00
./xmlchange --subgroup case.st_archive JOB_WALLCLOCK_TIME=01:00:00
./xmlchange STOP_OPTION="ndays"
./xmlchange STOP_N=10
./xmlchange RESUBMIT=1

./xmlchange CLM_ACCELERATED_SPINUP="sasu"
./xmlchange CLM_FORCE_COLDSTART=off

./case.setup
./preview_namelists
./case.build

### DO NOT DO ./case.submit: meristem.run submits the case ###
