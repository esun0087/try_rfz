import codecs
import os
import numpy as np
import get_true_pdb_name
import pickle
import torch

from ffindex import *
from kinematics import xyz_to_t2d
from parsers import parse_a3m, read_templates

def read_xyz(path):
    """
    x y z x y z x y z
    x y z x y z x y z
    """
    return np.load(path).astype(np.float)
def read_mask(path):
    data = np.load(path)
    return data.astype(np.int)

def read_dis_angle(path):
    """
    L * L * 4
    dist
    omega
    theta
    phi
    """
    data = np.load(path)
    dis_distribute = data[..., 0]
    omega_distribute = data[..., 1]
    theta_distribute = data[..., 2]
    phi_distribute = data[..., 3]
    return [dis_distribute.astype(np.int), omega_distribute.astype(np.int), theta_distribute.astype(np.int), phi_distribute.astype(np.int)]
def read_data_forsave(data_path):
    FFDB="pdb100_2021Mar03/pdb100_2021Mar03/pdb100_2021Mar03"
    FFindexDB = namedtuple("FFindexDB", "index, data")
    ffdb = None
    ffdb = FFindexDB(read_index(FFDB+'_pdb.ffindex'),
                     read_data(FFDB+'_pdb.ffdata'))
    data = []
    def check_file_ok(seq_feat_path, seq_name):
        files = ["t000_.msa0.a3m", "t000_.hhr", "t000_.atab", seq_name + ".xyz.npy", seq_name + ".dis_angle.npy", seq_name + ".mask.npy"]
        return all([os.path.exists(os.path.join(seq_feat_path, i)) for i in files])
    for line in codecs.open(data_path):
        # if len(data) >= 10:
        #     break
        line = line.strip().split(",")
        seq_name,seq_feat_path = line
        if not check_file_ok(seq_feat_path, seq_name):
            continue
        msa = parse_a3m(os.path.join(seq_feat_path, "t000_.msa0.a3m"))
        N, L = msa.shape
        # if L > 100:
        #     continue
        xyz_t, t1d, t0d = read_templates(L, ffdb, os.path.join(seq_feat_path, "t000_.hhr"), \
            os.path.join(seq_feat_path, "t000_.atab"), n_templ=10)
        if xyz_t is None:
            continue
        t2d = xyz_to_t2d(xyz_t, t0d)
        # print(seq_name,seq_feat_path)
        xyz_label = read_xyz(os.path.join(seq_feat_path, seq_name + ".xyz.npy"))
        prob_s_label_2 = read_dis_angle(os.path.join(seq_feat_path, seq_name + ".dis_angle.npy"))
        dis_masks = read_mask(os.path.join(seq_feat_path, seq_name + ".mask.npy"))
        label = []
        label.append(torch.from_numpy(xyz_label).float())
        label.extend([torch.from_numpy(i) for i in prob_s_label_2])

        masks = torch.from_numpy(dis_masks).long()
        feat = torch.from_numpy(msa).long(), xyz_t, t1d, t0d, t2d

        print(f"debug {seq_name} msa {torch.from_numpy(msa).shape} xyz_t {xyz_t.shape} \
            xyz_label {torch.from_numpy(xyz_label).shape}")
        data.append((feat, label, masks))

    del(ffdb)
    get_true_pdb_name.clear()
    print("data reader over")
    return data
def save_train2pickle():
    data = read_data_forsave("./generate_feat/train-feat.list")
    data_new = []
    for feat, label, masks in data:
        feat_new = []
        label_new = []
        masks_new = []
        for j, f in enumerate(feat):
            feat_new.append(f.tolist())
        
        for j, l in enumerate(label):
            label_new.append(l.tolist())

        for j, l in enumerate(masks):
            masks_new.append(l.tolist())
        data_new.append((feat_new, label_new, masks_new))
    with open("./generate_feat/train_data.pickle", "wb") as f:
        pickle.dump(data_new, f)

if __name__ == '__main__':
    save_train2pickle()