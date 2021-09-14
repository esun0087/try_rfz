import sys, os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import loss
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
import rigid_transform_3D
from torch.nn.utils import clip_grad_norm_
script_dir = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
from train_config import *

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

    def cross_loss_mask(self, pred_, true_, mask):
        sel = torch.where(mask == 1)
        true_ = true_[sel]
        pred_ = pred_[sel]
        cross_func = torch.nn.CrossEntropyLoss()
        loss = cross_func(pred_, true_)
        return loss

    def coords_loss_rotate(self,pred_, true_):
        mse_loss = torch.nn.MSELoss()
        losses = []
        for pred_true, cur_true in zip(pred_, true_):
            cur_mask = torch.where(~torch.isnan(cur_true))
            cur_true = cur_true[cur_mask].view(-1, 3,)
            pred_true = pred_true[cur_mask].view(-1, 3,)
            R, t = rigid_transform_3D.rigid_transform_3D2(pred_true, cur_true)
            pred_rotate = torch.matmul(pred_true, R) + t
            c = mse_loss(pred_rotate, cur_true)
            c = torch.sqrt(c)
            losses.append(c)
        return c

    def coords_loss_rotate_new(self,pred_, true_):
        def get_r_t(pred_, true_):
            true_ = true_.view(-1, 3, 3)
            pred_ = pred_.view(-1, 3, 3)
            true_ca = true_[:,1,:]
            pred_ca = pred_[:,1,:]
            R, t = rigid_transform_3D.rigid_transform_3D2(pred_ca, true_ca)
            return R, t
        losses = []
        mse_loss = torch.nn.MSELoss()
        for pred_true, cur_true in zip(pred_, true_):
            cur_mask = torch.where(~torch.isnan(cur_true))
            cur_true = cur_true[cur_mask].view(-1, 3,)
            pred_true = pred_true[cur_mask].view(-1, 3,)
            R, t = get_r_t(pred_true, cur_true)

            pred_rotate = torch.matmul(pred_true, R) + t
            c = mse_loss(pred_rotate, cur_true)
            c = torch.sqrt(c)
            losses.append(c)
        return sum(losses)/len(losses)

    def dis_mse_loss2(self, predicted_points, true_points):
        """
        compute whole matrix loss
        """
        # Compute true and predicted distance matrices.
        dmat_true = torch.sqrt(1e-10 + torch.sum((true_points[:, :, None] - true_points[:, None, :])**2, axis=-1))

        dmat_predicted = torch.sqrt(1e-10 + torch.sum(
                (predicted_points[:, :, None] -
                predicted_points[:, None, :])**2, axis=-1))
        sel = torch.where((~torch.isnan(dmat_true)).float())
        
        dmat_true = dmat_true[sel]
        dmat_predicted = dmat_predicted[sel]
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(dmat_predicted, dmat_true)
        loss = torch.sqrt(loss)
        return loss

    def dis_mse_loss(self, predicted_points, true_points):
        """
        is just like lddt
        """
        # Compute true and predicted distance matrices.
        dmat_true = torch.sqrt(1e-10 + torch.sum((true_points[:, :, None] - true_points[:, None, :])**2, axis=-1))

        dmat_predicted = torch.sqrt(1e-10 + torch.sum(
                (predicted_points[:, :, None] -
                predicted_points[:, None, :])**2, axis=-1))
        sel = torch.where(dmat_true < 15)

        dmat_true = dmat_true[sel]
        dmat_predicted = dmat_predicted[sel]
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(dmat_predicted, dmat_true)
        loss = torch.sqrt(loss)
        return loss
    def train_with_mask(self, data_path):
        train_data = data_reader.DataRead(data_path)
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=data_reader.collate_batch_data)
        # dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = lr_scheduler.MultiStepLR(optimizer, [500, 800], 0.1)
        epoch = 0
        while 1:
            avg_loss, data_cnt = 0, 0
            for i, data in enumerate(dataloader):
                if len(data) == 3:
                    feat, label, masks = data
                    lens_info = None
                else:
                    feat, label, masks, lens_info = data
                feat = [i.to(self.device) for i in feat]
                label = [i.to(self.device) for i in label]
                dis_mask = masks.to(self.device)
                optimizer.zero_grad()
                msa, xyz_t, t1d, t0d = feat
                xyz_label, dis_label, omega_label, theta_label, phi_label  = label
                xyz, lddt, prob_s = self.get_model_result(msa, xyz_t, t1d, t0d, lens_info = lens_info)
                dis_prob, omega_prob, theta_prob, phi_prob = prob_s
                batch_size = xyz_label.shape[0]
                # print("xyz label shape", xyz_label.shape, "xyz shape", xyz.shape)

                dis_loss = self.cross_loss_mask(dis_prob.float(), dis_label, dis_mask)
                # oemga_loss = self.cross_loss_mask(omega_prob.float(), omega_label, dis_mask)
                # theta_loss = self.cross_loss_mask(theta_prob.float(), theta_label, dis_mask)
                # phi_loss = self.cross_loss_mask(phi_prob.float(), phi_label, dis_mask)
                xyz = xyz.view(batch_size, -1, 3)
                xyz_loss = self.coords_loss_rotate_new(xyz.float(), xyz_label.float())
                # dis_loss_2 = self.dis_mse_loss2(xyz.float(), xyz_label.float())
                xyz_loss = xyz_loss
                lddt_result = lddt_torch.lddt(xyz.float(), xyz_label.float())

                loss = [\
                    # dis_loss, \
                    # oemga_loss, \
                    # theta_loss, \
                    # phi_loss, \
                    xyz_loss, \
                    # dis_loss_2
                    ]
                # print("loss ", ["%.3f" % i.data for i in loss], end = " ")
                sum_loss = sum(loss)
                avg_loss += sum_loss.cpu().detach().numpy()
                if 1:
                    for i, lo in enumerate(loss):
                        if i == len(loss) - 1:
                            lo.backward()
                        else:
                            lo.backward(retain_graph=True)
                clip_grad_norm_(self.model.parameters(), max_norm=3, norm_type=2)
                optimizer.step()
                data_cnt += 1
            scheduler.step()
            avg_loss = avg_loss / data_cnt
            print(f"=================train epoch {epoch} {'%.3f' % avg_loss} lddt {lddt_result}")
            epoch += 1

    def for_single(self, msa, t1d, t2d, lens_info = None):
        B, N, L = msa.shape
        idx_pdb = torch.arange(L).long().expand((B, L))
        msa = msa[:,:1000].to(self.device)
        seq = msa[:,0]
        idx_pdb = idx_pdb.to(self.device)
        t1d = t1d[:,:10].to(self.device)
        t2d = t2d[:,:10].to(self.device)

        logit_s, _, xyz, lddt = self.model(msa, seq, idx_pdb, t1d=t1d, t2d=t2d, lens_info=lens_info)
        logit_s = list(logit_s)
        for i, v in enumerate(logit_s):
            logit_s[i] = v.permute(0,2,3,1)
        return xyz, lddt, logit_s # 目前只知道距离的计算方法，还不知道角度的计算方法
    def get_model_result(self, msa, xyz_t, t1d, t0d, lens_info = None):
        B, N, L = msa.shape
        # msa = torch.tensor(msa).long()
        xyz_t = xyz_t.float()
        t1d = t1d.float()
        t0d = t0d.float()
        t2d = xyz_to_t2d(xyz_t, t0d)
       
        xyz, lddt, prob_s = self.for_single(msa, t1d, t2d, lens_info=lens_info)
        return xyz, lddt, prob_s

if __name__ == "__main__":
    train = Train(use_cpu=True)
    train.train_with_mask("./generate_feat/train_data.pickle")

