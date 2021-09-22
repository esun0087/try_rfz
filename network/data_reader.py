import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

def read_data_true_mask(data_path):
    f = open(data_path, "rb")
    data = pickle.load(f)
    data = data[1:]
    
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


def test_dataloader():
    train_data = DataRead("./generate_feat/train_data.pickle")
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=collate_batch_data)
    for i, data in enumerate(dataloader):
        # print(data)
        feat, label, masks = data
        print(len(feat[0]))
if __name__ == '__main__':
    test_dataloader()