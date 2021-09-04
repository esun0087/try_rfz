DATA_PATH=../train-data

ls pdb/* | head -n 500 > train-pdb.list
python3 generate_fasta.py $DATA_PATH
python3 generate_xyz.py $DATA_PATH 10
python3 generate_dis.py $DATA_PATH 10
python3 generate_ncaccb.py $DATA_PATH 10
python3 generate_dis_angle.py $DATA_PATH 10
# python3 generate_msa.py $DATA_PATH