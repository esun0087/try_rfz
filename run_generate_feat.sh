#!/bin/bash

# make the script stop when error (non-true exit code) is occured
set -e

############################################################
SCRIPT=`realpath -s $0`
export PIPEDIR=`dirname $SCRIPT`

CPU="8"  # number of CPUs to use
MEM="8" # max memory (in GB)

# Inputs:
IN="$1"                # input.fasta
WDIR=`realpath -s $2`  # working folder

LEN=`tail -n1 $IN | wc -m`

mkdir -p $WDIR/log

############################################################
# 1. generate MSAs
############################################################
if [ ! -s $WDIR/t000_.msa0.a3m ]
then
    echo "Running HHblits"
    $PIPEDIR/input_prep/make_msa.sh $IN $WDIR $CPU $MEM > $WDIR/log/make_msa.stdout 2> $WDIR/log/make_msa.stderr
fi

############################################################
# 3. search for templates
############################################################
DB="/mnt/e/study/RoseTTAFold-Train/data/uniclust30_2016_09/uniclust30_2016_09"
if [ ! -s $WDIR/t000_.hhr ]
then
    echo "Running hhsearch"
    HH="hhsearch -b 50 -B 500 -z 50 -Z 500 -mact 0.05 -cpu $CPU -maxmem $MEM -aliw 100000 -e 100 -p 5.0 -d $DB"
    cat $WDIR/t000_.msa0.a3m > $WDIR/t000_.msa0.ss2.a3m
    $HH -i $WDIR/t000_.msa0.ss2.a3m -o $WDIR/t000_.hhr -atab $WDIR/t000_.atab -v 0 > $WDIR/log/hhsearch.stdout 2> $WDIR/log/hhsearch.stderr
fi

############################################################
# 3. search for templates
############################################################
# DB="./uniclust30_2016_09/uniclust30_2016_09/uniclust30_2016_09"
# if [ ! -s $WDIR/t000_.hhr ]
# then
#     echo "Running hhsearch"
#     HH="hhsearch -B 500 -Z 500 -mact 0.05 -cpu $CPU -maxmem $MEM -aliw 100000 -e 10 -p 5.0 -d $DB"
#     cat $WDIR/t000_.msa0.a3m > $WDIR/t000_.msa0.ss2.a3m
#     $HH -i $WDIR/t000_.msa0.ss2.a3m -o $WDIR/t000_.hhr -atab $WDIR/t000_.atab -v 0 > $WDIR/log/hhsearch.stdout 2> $WDIR/log/hhsearch.stderr
# fi
# echo "Done"
