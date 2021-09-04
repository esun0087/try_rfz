#!/bin/bash

# inputs
in_fasta="$1"
out_dir="$2"

# resources
CPU="$3"
MEM="$4"

# sequence databases
# DB="/mnt/e/study/RoseTTAFold-Train/data/uniclust30_2016_09/uniclust30_2016_09/uniclust30_2016_09"
# DB="/mnt/e/study/RoseTTAFold-Train/data/pdb70/pdb70"
DB="/mnt/f/pdb100_2021Mar03/pdb100_2021Mar03/pdb100_2021Mar03"
MYDB="$PIPEDIR/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"

# setup hhblits command
HHBLITS="hhblits -o /dev/null -mact 0.35 -maxfilt 100000000 -neffmax 20 -cov 25 -cpu 16 -nodiff -realign_max 100000000 -maxseq 1000000 -maxmem $MEM -n 1 -d $DB "
echo $HHBLITS

mkdir -p $out_dir/hhblits
tmp_dir="$out_dir/hhblits"
out_prefix="$out_dir/t000_"

# perform iterative searches
prev_a3m="$in_fasta"
for e in 10
do
    echo $e
    # $HHBLITS -i $prev_a3m -oa3m $tmp_dir/t000_.$e.a3m -e $e -v 0
    $HHBLITS -i $prev_a3m -o ${out_prefix}.hhr -oa3m ${out_prefix}.msa0.a3m -atab ${out_prefix}.atab -e $e -v 0 
    hhfilter -id 90 -cov 75 -i $tmp_dir/t000_.$e.a3m -o $tmp_dir/t000_.$e.id90cov75.a3m
    hhfilter -id 90 -cov 50 -i $tmp_dir/t000_.$e.a3m -o $tmp_dir/t000_.$e.id90cov50.a3m
    prev_a3m="$tmp_dir/t000_.$e.id90cov50.a3m"
    n75=`grep -c "^>" $tmp_dir/t000_.$e.id90cov75.a3m`
    n50=`grep -c "^>" $tmp_dir/t000_.$e.id90cov50.a3m`

    if ((n75>2000))
    then
        if [ ! -s ${out_prefix}.msa0.a3m ]
        then
            cp $tmp_dir/t000_.$e.id90cov75.a3m ${out_prefix}.msa0.a3m
	    break
        fi
    elif ((n50>4000))
    then
        if [ ! -s ${out_prefix}.msa0.a3m ]
        then
            cp $tmp_dir/t000_.$e.id90cov50.a3m ${out_prefix}.msa0.a3m
            break
        fi
    else
        continue
    fi

done

if [ ! -s ${out_prefix}.msa0.a3m ]
then
    cp $tmp_dir/t000_.1e-3.id90cov50.a3m ${out_prefix}.msa0.a3m
fi
