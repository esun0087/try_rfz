DATA_PATH=../train-data

#python3 split_pdb.py $DATA_PATH 10
python3 generate_fasta.py $DATA_PATH 1
python3 generate_xyz.py $DATA_PATH 1
# python3 generate_dis.py $DATA_PATH 1
python3 generate_ncaccb.py $DATA_PATH 1
python3 generate_dis_angle.py $DATA_PATH 1
# python3 generate_msa.py $DATA_PATH