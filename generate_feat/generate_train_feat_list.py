import codecs
import os
import sys

def check_file_ok(seq_feat_path, seq_name):
    files = ["t000_.msa0.a3m", "t000_.hhr", "t000_.atab", seq_name + ".xyz.npy", seq_name + ".dis_angle.npy", seq_name + ".mask.npy"]
    return all([os.path.exists(os.path.join(seq_feat_path, i)) for i in files])
if __name__ == '__main__':
    base_data_dir = sys.argv[1]
    fout = codecs.open("train-feat.list", "w", "utf-8")
    for line in codecs.open("train-pdb.list", "r", "utf-8"):
        line = line.strip()
        pdb_name = line.split('/')[-1].split(".")[0]
        feat_dir = os.path.join(base_data_dir, pdb_name)
        feat_path = os.path.join(feat_dir, "t000_.msa0.a3m")
        feat2 = os.path.join(feat_dir, pdb_name + ".xyz.npy")
        if check_file_ok(feat_dir, pdb_name):
            fout.write(pdb_name + "," + feat_dir + "\n")
    fout.close()


