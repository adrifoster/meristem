#!/usr/bin/env bash
#
# tack-on script for meristem ensemble example

if [ $# -lt 1 ]; then
  echo "ERROR: please specify a case_root"
  exit 1
fi
case_root="$1"

cd ${case_root}
./xmlchange STOP_N=2
./xmlchange STOP_OPTION='nyears'
./xmlchange RESUBMIT=0
