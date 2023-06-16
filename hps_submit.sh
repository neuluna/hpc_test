#!/bin/bash
var="bagls2"
parameter="_test"
jobname="${var}${parameter}"

sbatch --job-name=$jobname --time=24:00:00 datapruning/datapruning.sh $var 

    