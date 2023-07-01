#!/bin/bash
var="bagls"
parameter="_test"
jobname="${var}${parameter}"

sbatch --job-name=$jobname --time=24:00:00 hpc_test/hpc-job.sh $var 
