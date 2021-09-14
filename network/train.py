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
import rigid_transform_3D
from torch.nn.utils import clip_grad_norm_
from multi_backward import MultiBackward
script_dir = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
from train_config import *
torch.autograd.set_detect_anomaly(True)
cur_R, cur_t = None, None

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
        pred_ = pred_.reshape(-1, pred_.shape[-1])
        true_ = torch.flatten(true_)
        mask = torch.flatten(mask)
        cross_func = torch.nn.CrossEntropyLoss(reduction='none')
        pred_ = cross_func(pred_, true_)
        mask_loss = mask * pred_
        result = torch.mean(mask_loss)
        return result

    def coords_loss(self,pred_, true_):
        mse_loss = torch.nn.MSELoss()
        cur_mask = (torch.isnan(true_)).float()
        cur_mask = 1 - cur_mask
        true_ = cur_mask * true_
        pred_ = cur_mask * pred_
        c = mse_loss(pred_, true_)
        c = torch.sqrt(c)
        return c

    def coords_loss_rotate(self,pred_, true_):
        mse_loss = torch.nn.MSELoss()
        cur_mask = (torch.isnan(true_)).float()
        cur_mask = 1 - cur_mask
        true_ = cur_mask * true_
        pred_ = cur_mask * pred_
        R, t = rigid_transform_3D.rigid_transform_3D2(pred_[0], true_[0])
        pred_rotate = torch.matmul(pred_, R) + t
        c = mse_loss(pred_rotate, true_)
        c = torch.sqrt(c)
        return c

    def coords_loss_rotate_new(self,pred_, true_):
        def get_r_t(pred_, true_):
            global cur_R
            global cur_t
            if cur_R is not None:
                return cur_R, cur_t
            true_ = true_.view(-1, 3, 3)
            pred_ = pred_.view(-1, 3, 3)
            true_ca = true_[:,1,:]
            pred_ca = pred_[:,1,:]
            select_atoms = torch.where(~torch.isnan(true_ca))

            true_coords = true_ca[select_atoms].view(-1, 3)
            pred_coords = pred_ca[select_atoms].view(-1, 3)
            
            R, t = rigid_transform_3D.rigid_transform_3D2(pred_coords, true_coords)
            cur_R = R
            cur_t = t
            return R, t
        losses = 0
        for pred_true, cur_true in zip(pred_, true_):
            mse_loss = torch.nn.MSELoss()
            cur_mask = (~torch.isnan(true_)).float()
            true_ = cur_mask * true_
            pred_ = cur_mask * pred_
            R, t = get_r_t(pred_true, cur_true)
            pred_rotate = torch.matmul(pred_, R) + t
            c = mse_loss(pred_rotate, true_)
            c = torch.sqrt(c)
            losses = losses + c
        return losses

    def dis_cos_loss(self, predicted_points, true_points):
        """
        compute whole matrix loss
        """
        # Compute true and predicted distance matrices.
        dmat_true = torch.sqrt(1e-10 + torch.sum((true_points[:, :, None] - true_points[:, None, :])**2, axis=-1))

        dmat_predicted = torch.sqrt(1e-10 + torch.sum(
                (predicted_points[:, :, None] -
                predicted_points[:, None, :])**2, axis=-1))
        mask = (dmat_true < 15).float()
        dmat_true = mask * dmat_true
        dmat_predicted = mask * dmat_predicted
        tmp = torch.sum(dmat_true * dmat_predicted)
        normx = torch.sqrt(torch.sum(dmat_true * dmat_true)) * torch.sqrt(torch.sum(dmat_predicted * dmat_predicted))
        print(tmp, normx)
        return 1 - tmp / normx
    def dis_mse_loss2(self, predicted_points, true_points):
        """
        compute whole matrix loss
        """
        # Compute true and predicted distance matrices.
        dmat_true = torch.sqrt(1e-10 + torch.sum((true_points[:, :, None] - true_points[:, None, :])**2, axis=-1))

        dmat_predicted = torch.sqrt(1e-10 + torch.sum(
                (predicted_points[:, :, None] -
                predicted_points[:, None, :])**2, axis=-1))
        mask = (~torch.isnan(dmat_true)).float()
        
        dmat_true = mask * dmat_true
        dmat_predicted = mask * dmat_predicted
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
        mask = (dmat_true < 15).float()

        dmat_true = mask * dmat_true
        dmat_predicted = mask * dmat_predicted
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(dmat_predicted, dmat_true)
        loss = torch.sqrt(loss)
        return loss
    def train_with_mask(self, data_path):
        train_data = data_reader.DataRead(data_path)
        # dataloader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=data_reader.collate_batch_data)
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = lr_scheduler.MultiStepLR(optimizer, [500, 800], 0.1)
        epoch = 0
        
        while 1:
            avg_loss, data_cnt = 0, 0
            multi_back = MultiBackward(optimizer, 1)
            for batch_idx, data in enumerate(dataloader):
                feat, label, masks = data
                feat = [i.to(self.device) for i in feat]
                label = [i.to(self.device) for i in label]
                dis_mask = masks.to(self.device)
                msa, xyz_t, t1d, t0d = feat
                xyz_label, dis_label, omega_label, theta_label, phi_label  = label
                xyz, lddt, prob_s = self.get_model_result(msa, xyz_t, t1d, t0d)
                dis_prob, omega_prob, theta_prob, phi_prob = prob_s
                batch_size = xyz_label.shape[0]
                # print("xyz label shape", xyz_label.shape, "xyz shape", xyz.shape)

                # dis_loss = self.cross_loss_mask(dis_prob.float(), dis_label, dis_mask)
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
                print("all loss ", ["%.3f" % i.data for i in loss], end = " ")
                sum_loss = sum(loss)
                multi_back.add_loss(sum_loss)
                avg_loss += sum_loss.cpu().detach().numpy()
                # clip_grad_norm_(self.model.parameters(), max_norm=3, norm_type=2)
                data_cnt += 1
            del (multi_back)
            
            scheduler.step()
            avg_loss = avg_loss / data_cnt
            print(f"=================train epoch {epoch} avg_loss {avg_loss} lddt {lddt_result}")
            epoch += 1

    def for_single(self, msa, t0d, t1d, t2d):
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
       
        xyz, lddt, prob_s = self.for_single(msa, t0d, t1d, t2d)
        return xyz, lddt, prob_s

if __name__ == "__main__":
    train = Train(use_cpu=True)
    train.train_with_mask("./generate_feat/train_data.pickle")

