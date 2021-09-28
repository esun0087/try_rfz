import sys, os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
from torch.utils import data
from RoseTTAFoldModel  import RoseTTAFoldModule_e2e
from collections import namedtuple
from ffindex import *
from kinematics import xyz_to_t2d
from trFold import TRFold
import torch.optim as optim
from torch.optim import lr_scheduler
import data_reader
import lddt_torch
from torch.nn.utils import clip_grad_value_
from multi_backward import MultiBackward
from loss import Loss
import time
script_dir = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
from train_config import *
torch.autograd.set_detect_anomaly(True)

def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
class Train():
    def __init__(self, use_cpu=False):
        #
        # define model name
        self.model_name = "RoseTTAFold"
        if torch.cuda.is_available() and (not use_cpu):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.active_fn = nn.Softmax(dim=1)

        # define model & load model
        self.model = RoseTTAFoldModule_e2e(**MODEL_PARAM).to(self.device)
        self.loss = Loss(self.device)

    def train_with_mask(self, data_path):
        train_data = data_reader.DataRead(data_path)
        # dataloader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=data_reader.collate_batch_data)
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = lr_scheduler.MultiStepLR(optimizer, [500, 800], 0.1)
        epoch_max = 2000
        
        for epoch in range(epoch_max):
            avg_loss, data_cnt = 0, 0
            multi_back = MultiBackward(optimizer, 1)
            weight = (epoch + 1) / epoch_max * 0.2 + 0.05
            for batch_idx, data in enumerate(dataloader):
                feat, label, masks = data
                feat = [i.to(self.device) for i in feat]
                label = [i.to(self.device) for i in label]
                dis_mask = masks.to(self.device)
                msa, xyz_t, t1d, t0d = feat
                xyz_label, dis_label, omega_label, theta_label, phi_label  = label
                xyz, model_lddt, prob_s = self.get_model_result(msa, xyz_t, t1d, t0d)
                dis_prob, omega_prob, theta_prob, phi_prob = prob_s
                batch_size = xyz_label.shape[0]

                dis_loss = self.loss.cross_loss_mask(dis_prob.float(), dis_label, dis_mask)
                oemga_loss = self.loss.cross_loss_mask(omega_prob.float(), omega_label, dis_mask)
                theta_loss = self.loss.cross_loss_mask(theta_prob.float(), theta_label, dis_mask)
                phi_loss = self.loss.cross_loss_mask(phi_prob.float(), phi_label, dis_mask)

                xyz = xyz.view(batch_size, -1, 3)
                xyz_loss = self.loss.coords_loss_rotate(xyz.float(), xyz_label.float())
                dis_loss_whole = self.loss.dis_mse_whole_atom(xyz.float(), xyz_label.float())

                xyz_ca = xyz.view(batch_size, -1, 3, 3)[:,:,1]
                xyz_label_ca = xyz_label.view(batch_size, -1, 3, 3)[:,:,1]
                lddt_result = lddt_torch.lddt(xyz_ca.float(), xyz_label_ca.float())

                lddt_loss = self.loss.lddt_loss(xyz.float(), xyz_label.float(), model_lddt)
                loss = [\
                    dis_loss, \
                    oemga_loss, \
                    theta_loss, \
                    phi_loss, \
                    weight * xyz_loss, \
                    weight * dis_loss_whole, \
                    # dis_loss_ca, \
                    lddt_loss
                    ]
                print("all loss ", ["%.2f" % i.data for i in loss], "weight", weight)
                sum_loss = sum(loss)
                clip_grad_value_(self.model.parameters(), 1)
                multi_back.add_loss(sum_loss)
                avg_loss += sum_loss.cpu().detach().numpy()
                data_cnt += 1
            clip_grad_value_(self.model.parameters(), 1)
            del (multi_back)
            
            scheduler.step()
            avg_loss = avg_loss / data_cnt
            print("time is ", get_time(), end = " ")
            print(f"=====train epoch {epoch} avg_loss {avg_loss} lddt {lddt_result} model lddt {torch.mean(model_lddt)}")

    def for_single(self, msa, t1d, t2d):
        B, N, L = msa.shape
        idx_pdb = torch.arange(L).long().expand((B, L))
        msa = msa[:,:1000].to(self.device)
        seq = msa[:,0]
        idx_pdb = idx_pdb.to(self.device)
        t1d = t1d[:,:10].to(self.device)
        t2d = t2d[:,:10].to(self.device)

        logit_s, _, xyz, lddt = self.model(msa, seq, idx_pdb, t1d=t1d, t2d=t2d)
        logit_s = list(logit_s)
        for i, v in enumerate(logit_s):
            logit_s[i] = v.permute(0,2,3,1)
        return xyz, lddt, logit_s # 目前只知道距离的计算方法，还不知道角度的计算方法
    def get_model_result(self, msa, xyz_t, t1d, t0d, window=150, shift=75):
        B, N, L = msa.shape
        # msa = torch.tensor(msa).long()
        xyz_t = xyz_t.float()
        t1d = t1d.float()
        t0d = t0d.float()
        t2d = xyz_to_t2d(xyz_t, t0d)
       
        xyz, lddt, prob_s = self.for_single(msa, t1d, t2d)
        return xyz, lddt, prob_s

if __name__ == "__main__":
    train = Train(use_cpu=True)
    train.train_with_mask("./generate_feat/train_data.pickle")

