import enum
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from ffindex import *
import codecs
from parsers import parse_a3m, read_templates
import os
import get_true_pdb_name
import pickle
def read_data_mock(data_path):
    data = []
    for i in range(10):
        L = random.randint(80, 120)
        N = 10
        msa = torch.randint(0, 21, (N, L))
        xyz_t = torch.randn(N, L, 3, 3)
        t1d = torch.randn(N, L, 3)
        t0d = torch.randn(N,3)
        xyz_label =  torch.randn(L, 3, 3)
        prob_s_label = torch.randint(0, 64, (L * L, ))

        feat = msa, xyz_t, t1d, t0d
        label = xyz_label, prob_s_label
        data.append((feat, label))
    return data
def read_xyz(path):
    """
    x y z x y z x y z
    x y z x y z x y z
    """
    return np.load(path).astype(np.float)

def get_dis_class(v):
    segs = np.arange(2.5, 20.5, 0.5)
    for i,x in enumerate(segs):
        if i == 0 and v < x:
            return i
        if v < x and v > segs[i-1]:
            return i
    return len(segs)
def read_dis(path):
    """
    L * L
    """
    data = np.load(path)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j] = get_dis_class(data[i][j])
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

def read_mask(path):
    data = np.load(path)
    return data.astype(np.int)


def read_data_true_test(data_path):
    data = []
    for line in codecs.open(data_path):
        if len(data) >= 1:
            break
        line = line.strip().split(",")
        seq_name,seq_feat_path = line
        msa = parse_a3m(os.path.join(seq_feat_path, "t000_.msa0.a3m"))
        N, L = msa.shape
        if L > 50:
            continue
        xyz_t = torch.randn(N, L, 3, 3)
        t1d = torch.randn(N, L, 3)
        t0d = torch.randn(N,3)
        xyz_label = read_xyz(os.path.join(seq_feat_path, seq_name + ".xyz.npy"))
        prob_s_label = read_dis(os.path.join(seq_feat_path, seq_name + ".dis.npy"))
        label = torch.from_numpy(xyz_label).float(), torch.from_numpy(prob_s_label).long()
        feat = torch.from_numpy(msa).long(), xyz_t, t1d, t0d
        data.append((feat, label))
        # break

    print("data reader over")
    return data
def read_data_true(data_path):
    f = open(data_path, "rb")
    data = pickle.load(f)
    data = data[2:]
    
    train_data = []
    for feat, label in data:
        feat_new = [torch.tensor(f).float() for f in feat]
        feat_new[0] = feat_new[0].long()
        print(f"data shape {feat_new[0].shape}", )
        label_new = [torch.tensor(label[0]).float(), torch.tensor(label[1]).long()]
        train_data.append((tuple(feat_new), tuple(label_new)))
        if len(train_data) == 1:
            break
    f.close()
    print("data reader over")
    return train_data
def read_data_true_mask(data_path):
    f = open(data_path, "rb")
    data = pickle.load(f)
    # data = data[2:]
    
    train_data = []
    for feat, label, masks in data:
        # msa, xyz_t, t1d, t0d = feat
        feat_new = [torch.tensor(f).float() for f in feat]
        feat_new[0] = feat_new[0].long()
        # print(f"data shape {feat_new[0].shape}", )
        label_new = []
        label_new.append(torch.tensor(label[0]).float()) # xyz
        label_new.extend([torch.tensor(i).long() for i in label[1:]])

        masks_new = torch.tensor(masks).long()
        train_data.append((tuple(feat_new), tuple(label_new), masks_new))
    f.close()
    print("data reader over")
    return train_data

def collate_batch_data(batch_dic):
    max_seq_length = max([feat[0].shape[1] for feat, label, masks in batch_dic]) # 一批数据中最长的那个样本长度
    max_msa_len = max([feat[0].shape[0] for feat, label, masks in batch_dic]) # 一批数据中最长的那个样本长度
    # print(f"collate_batch_data max_msa_len {max_msa_len} max_seq_length {max_seq_length}")

    def feat_extend(feat):
        msa, xyz_t, t1d, t0d = feat

        msa_new = np.full((max_msa_len, max_seq_length), 20)
        msa_new[:msa.shape[0], :msa.shape[1]] = msa

        xyz_new = np.full((10, max_seq_length, 3, 3), np.nan)
        xyz_new[:,:xyz_t.shape[1]] = xyz_t

        t1d_new = np.full((10, max_seq_length, 3), np.nan)
        t1d_new[:,:t1d.shape[1]] = t1d

        t0d = t0d.numpy()

        return (msa_new, xyz_new, t1d_new, t0d)
    def label_extend(labels):
        """
        xyz
        dis
        oemga
        theta
        phi
        """
        xyz_label, dis_label, omege_label, theta_label, phi_label = labels
        cur_seq_len = omege_label.shape[0]
        # print("debug", xyz_label.shape,omege_label.shape)

        xyz_new_label = np.full((max_seq_length * 3, 3), np.nan)
        xyz_new_label[:cur_seq_len * 3] = xyz_label

        dis_new_label = np.full((max_seq_length, max_seq_length), 0)
        dis_new_label[:cur_seq_len, :cur_seq_len] = dis_label 

        omega_new_label = np.full((max_seq_length, max_seq_length), 0)
        omega_new_label[:cur_seq_len, :cur_seq_len] = omege_label 

        theta_new_label = np.full((max_seq_length, max_seq_length), 0)
        theta_new_label[:cur_seq_len, :cur_seq_len] = theta_label

        phi_new_label = np.full((max_seq_length, max_seq_length), 0)
        phi_new_label[:cur_seq_len, :cur_seq_len] = phi_label
        return [xyz_new_label, dis_new_label, omega_new_label, theta_new_label, phi_new_label]

    def masks_extend(masks):
        mask_new = np.full((max_seq_length, max_seq_length), 0.0)
        cur_seq_len = masks.shape[0]
        mask_new[:cur_seq_len, :cur_seq_len] = masks 
        return mask_new

    def split_feats(feats):
        msas = []
        xyz_ts = []
        t1ds = []
        t0ds = []
        for msa, xyz_t, t1d, t0d in feats:
            msas.append(msa)
            xyz_ts.append(xyz_t)
            t1ds.append(t1d)
            t0ds.append(t0d)
        msas = torch.tensor(msas).long()
        xyz_ts = torch.tensor(xyz_ts).float()
        t1ds = torch.tensor(t1ds).float()
        t0ds = torch.tensor(t0ds).float()
        return msas, xyz_ts, t1ds, t0ds
    def split_labels(labels):
        xyz_labels, dis_labels, omege_labels, theta_labels, phi_labels = [],[],[],[],[]
        for xyz_label, dis_label, omege_label, theta_label, phi_label in labels:
            xyz_labels.append(xyz_label)
            dis_labels.append(dis_label)
            omege_labels.append(omege_label)
            theta_labels.append(theta_label)
            phi_labels.append(phi_label)

        xyz_labels = torch.tensor(xyz_labels).float()
        dis_labels = torch.tensor(dis_labels).long()
        omege_labels = torch.tensor(omege_labels).long()
        theta_labels = torch.tensor(theta_labels).long()
        phi_labels = torch.tensor(phi_labels).long()

        return xyz_labels, dis_labels, omege_labels, theta_labels, phi_labels

    feat_batch=[]
    label_batch=[]
    masks_batch=[]
    for i in range(len(batch_dic)): 
        feat, label, masks = batch_dic[i]
        # msa, xyz_t, t1d, t0d = feat
        feat = feat_extend(feat)
        label = label_extend(label)

        feat_batch.append(feat)
        label_batch.append(label)
        masks = masks_extend(masks)
        masks_batch.append(masks)
    feat_batch = split_feats(feat_batch)
    label_batch = split_labels(label_batch)
    masks_batch = torch.tensor(masks_batch)
    return feat_batch, label_batch, masks_batch
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
        # print(seq_name,seq_feat_path)
        xyz_label = read_xyz(os.path.join(seq_feat_path, seq_name + ".xyz.npy"))
        prob_s_label_2 = read_dis_angle(os.path.join(seq_feat_path, seq_name + ".dis_angle.npy"))
        dis_masks = read_mask(os.path.join(seq_feat_path, seq_name + ".mask.npy"))
        label = []
        label.append(torch.from_numpy(xyz_label).float())
        label.extend([torch.from_numpy(i) for i in prob_s_label_2])

        masks = torch.from_numpy(dis_masks).long()
        feat = torch.from_numpy(msa).long(), xyz_t, t1d, t0d

        print(f"debug {seq_name} msa {torch.from_numpy(msa).shape} xyz_t {xyz_t.shape} \
            xyz_label {torch.from_numpy(xyz_label).shape}")
        data.append((feat, label, masks))

    del(ffdb)
    get_true_pdb_name.clear()
    print("data reader over")
    return data
class DataRead(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        # self.data = read_data_true(data_path)
        self.data = read_data_true_mask(data_path)
        pass
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


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

def test_dataloader():
    train_data = DataRead("./generate_feat/train_data.pickle")
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=collate_batch_data)
    for i, data in enumerate(dataloader):
        # print(data)
        feat, label, masks = data
        print(len(feat[0]))
if __name__ == '__main__':
    save_train2pickle()
    # test_dataloader()