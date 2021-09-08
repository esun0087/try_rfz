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
script_dir = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])

NBIN = [37, 37, 37, 19]

# 参数都被缩小了，太大了本机跑不动
MODEL_PARAM ={
        "n_module"     : 1, # IterativeFeatureExtractor使用，用于迭代更新msa和pair的相互参考信息
        "n_module_str" : 1, # IterativeFeatureExtractor使用， 用于se3更新坐标和msa以及pair的信息,这是第一次更新。
        "n_module_ref" : 1, # Refine_module 使用， 也是为了se3更新坐标和msa以及pair的信息， 这个要迭代200次， 但是为了测试， 只迭代1次,这个过程中topk为64
        "n_layer"      : 1, # IterativeFeatureExtractor使用，
        "d_msa"        : 8 ,
        "d_pair"       : 8,
        "d_templ"      : 8,
        "n_head_msa"   : 2,
        "n_head_pair"  : 2,
        "n_head_templ" : 2,
        "d_hidden"     : 8,
        "r_ff"         : 4,
        "n_resblock"   : 1,
        "p_drop"       : 0.0,
        "use_templ"    : True,
        # "performer_N_opts": {"nb_features": 16},
        # "performer_L_opts": {"nb_features": 16}
        }

SE3_param = {
        "num_layers"    : 1,
        "num_channels"  : 8,
        "num_degrees"   : 2,
        "l0_in_features": 8,
        "l0_out_features": 8,
        "l1_in_features": 3,
        "l1_out_features": 3,
        "num_edge_features": 8,
        "div": 2,
        "n_heads": 1
        }

REF_param = {
        "num_layers"    : 1,
        "num_channels"  : 8,
        "num_degrees"   : 3,
        "l0_in_features": 8,
        "l0_out_features": 8,
        "l1_in_features": 3,
        "l1_out_features": 3,
        "num_edge_features": 8,
        "div": 4,
        "n_heads": 1
        }
MODEL_PARAM['SE3_param'] = SE3_param
MODEL_PARAM['REF_param'] = REF_param

# params for the folding protocol
fold_params = {
    "SG7"     : np.array([[[-2,3,6,7,6,3,-2]]])/21,
    "SG9"     : np.array([[[-21,14,39,54,59,54,39,14,-21]]])/231,
    "DCUT"    : 19.5,
    "ALPHA"   : 1.57,
    
    # TODO: add Cb to the motif
    "NCAC"    : np.array([[-0.676, -1.294,  0.   ],
                          [ 0.   ,  0.   ,  0.   ],
                          [ 1.5  , -0.174,  0.   ]], dtype=np.float32),
    "CLASH"   : 2.0,
    "PCUT"    : 0.5,
    "DSTEP"   : 0.5,
    "ASTEP"   : np.deg2rad(10.0),
    "XYZRAD"  : 7.5,
    "WANG"    : 0.1,
    "WCST"    : 0.1
}

fold_params["SG"] = fold_params["SG9"]

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
        R, t = rigid_transform_3D.rigid_transform_3D2(pred_, true_)
        pred_rotate = torch.matmul(pred_, R) + t
        c = mse_loss(pred_rotate, true_)
        c = torch.sqrt(c)
        return c

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
    def train_with_mask_v2(self, data_path):
        train_data = data_reader.DataRead(data_path)
        # dataloader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=data_reader.collate_batch_data)
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        scheduler = lr_scheduler.MultiStepLR(optimizer, [200, 500, 800], 0.1)
        epoch = 0
        while 1:
            avg_loss, data_cnt = 0, 0
            for i, data in enumerate(dataloader):
                feat, label, masks = data
                feat = [i.to(self.device) for i in feat]
                label = [i.to(self.device) for i in label]
                dis_mask = masks.to(self.device)
                optimizer.zero_grad()
                msa, xyz_t, t1d, t0d = feat
                xyz_label, dis_label, omega_label, theta_label, phi_label  = label
                xyz, lddt, prob_s = self.get_model_result(msa, xyz_t, t1d, t0d)
                dis_prob, omega_prob, theta_prob, phi_prob = prob_s
                batch_size = xyz_label.shape[0]
                # print("xyz label shape", xyz_label.shape, "xyz shape", xyz.shape)

                dis_loss = self.cross_loss_mask(dis_prob.float(), dis_label, dis_mask)
                oemga_loss = self.cross_loss_mask(omega_prob.float(), omega_label, dis_mask)
                theta_loss = self.cross_loss_mask(theta_prob.float(), theta_label, dis_mask)
                phi_loss = self.cross_loss_mask(phi_prob.float(), phi_label, dis_mask)
                xyz = xyz.view(batch_size, -1, 3)
                xyz_loss = self.coords_loss_rotate(xyz.float(), xyz_label.float())
                dis_loss_2 = self.dis_mse_loss(xyz.float(), xyz_label.float())
                xyz_loss = xyz_loss
                lddt_result = lddt_torch.lddt(xyz.float(), xyz_label.float())

                loss = [\
                    # dis_loss, \
                    # oemga_loss, \
                    # theta_loss, \
                    # phi_loss, \
                    xyz_loss, \
                    dis_loss_2
                    ]
                print("all loss ", ["%.3f" % i.data for i in loss], end = " ")
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
            print(f"=================train epoch {epoch} avg_loss {avg_loss} lddt {lddt_result}")
            epoch += 1

    def train_with_mask(self, data_path):
        torch.autograd.set_detect_anomaly(True)
        train_data = data_reader.DataRead(data_path)
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = lr_scheduler.MultiStepLR(optimizer, [200, 500], 0.1)
        cross_loss = nn.CrossEntropyLoss()
        mse_loss = torch.nn.MSELoss()
        epoch = 0
        while 1:
            avg_loss, data_cnt = 0, 0
            for i, data in enumerate(dataloader):
                feat, label, masks = data
                feat = [i.to(self.device) for i in feat]
                label = [i.to(self.device) for i in label]
                masks = [i.to(self.device) for i in masks]
                dis_mask, xyz_mask = masks
                sel_xyz = torch.where(xyz_mask == 1)
                optimizer.zero_grad()
                msa, xyz_t, t1d, t0d = feat
                xyz_label, dis_label, omega_label, theta_label, phi_label  = label
                xyz, lddt, prob_s = self.get_model_result(msa, xyz_t, t1d, t0d)
                # print(f"train prob_s {prob_s.shape} prob_s_label {prob_s_label.shape} xyz {xyz.shape} xyz_label {xyz_label.shape}")
                sel_pos = torch.where(dis_mask == 1)
                # prob_s_label = torch.flatten(prob_s_label)
                dis_prob, omega_prob, theta_prob, phi_prob = prob_s

                dis_prob = dis_prob[sel_pos]
                omega_prob = omega_prob[sel_pos]
                theta_prob = theta_prob[sel_pos]
                phi_prob = phi_prob[sel_pos]

                dis_label = dis_label[sel_pos]
                omega_label = omega_label[sel_pos]
                theta_label = theta_label[sel_pos]
                phi_label = phi_label[sel_pos]

                dis_loss = cross_loss(dis_prob.float(), dis_label)
                oemga_loss = cross_loss(omega_prob.float(), omega_label)
                theta_loss = cross_loss(theta_prob.float(), theta_label)
                phi_loss = cross_loss(phi_prob.float(), phi_label)

                xyz = xyz[sel_xyz]
                xyz_label = xyz_label[sel_xyz]

                xyz_label = xyz_label.view(-1, 3 * 3)
                xyz = xyz.view(-1, 3 * 3)
                mse_loss_sum = mse_loss(xyz.float(), xyz_label.float())
                mse_loss_sum = torch.sqrt(mse_loss_sum)

                # lddt_sum = lddt_torch.lddt(xyz.view(1, -1, 3).float(), xyz_label.view(1, -1, 3).float(), False)
                # print(f"lddt is {lddt_sum}")

                # loss = cross_loss_sum + mse_loss_sum
                # loss = [mse_loss_sum]
                loss = [\
                    # dis_loss, \
                    # oemga_loss, \
                    # theta_loss, \
                    phi_loss\
                    ]
                sum_loss = sum(loss)
                avg_loss += sum_loss.cpu().detach().numpy()
                # loss.backward()
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
            print(f"=================train epoch {epoch} avg_loss {avg_loss} lddt {torch.sum(lddt)}")
            epoch += 1
    def for_cropped(self, msa, t0d, t1d, t2d, window, shift):
        B, N, L = msa.shape
        idx_pdb = torch.arange(L).long().view(1, L)
        prob_s = [np.zeros((L,L,NBIN[i]), dtype=np.float32) for  i in range(4)]
        count_1d = np.zeros((L,), dtype=np.float32)
        count_2d = np.zeros((L,L), dtype=np.float32)
        node_s = np.zeros((L,MODEL_PARAM['d_msa']), dtype=np.float32)
        #
        grids = np.arange(0, L-window+shift, shift)
        ngrids = grids.shape[0]
        print("ngrid:     ", ngrids)
        print("grids:     ", grids)
        print("windows:   ", window)

        for i in range(ngrids):
            for j in range(i, ngrids):
                start_1 = grids[i]
                end_1 = min(grids[i]+window, L)
                start_2 = grids[j]
                end_2 = min(grids[j]+window, L)
                sel = np.zeros((L)).astype(np.bool)
                sel[start_1:end_1] = True
                sel[start_2:end_2] = True
                
                input_msa = msa[:,:,sel]
                mask = torch.sum(input_msa==20, dim=-1) < 0.5*sel.sum() # remove too gappy sequences
                input_msa = input_msa[mask].unsqueeze(0)
                input_msa = input_msa[:,:1000].to(self.device)
                input_idx = idx_pdb[:,sel].to(self.device)
                input_seq = input_msa[:,0].to(self.device)
                #
                # Select template
                input_t1d = t1d[:,:,sel].to(self.device) # (B, T, L, 3)
                input_t2d = t2d[:,:,sel][:,:,:,sel].to(self.device)
                #
                print ("running crop: %d-%d/%d-%d"%(start_1, end_1, start_2, end_2), input_msa.shape)
                with torch.cuda.amp.autocast():
                    logit_s, node, init_crds, pred_lddt = self.model(input_msa, input_seq, input_idx, t1d=input_t1d, t2d=input_t2d, return_raw=True)
                #
                # Not sure How can we merge init_crds.....
                sub_idx = input_idx[0].cpu()
                sub_idx_2d = np.ix_(sub_idx, sub_idx)
                count_2d[sub_idx_2d] = count_2d[sub_idx_2d] + 1.0
                count_1d[sub_idx] = count_1d[sub_idx] + 1.0
                node_s[sub_idx] = node_s[sub_idx] + node[0].cpu().detach().numpy()
                for i_logit, logit in enumerate(logit_s):
                    prob = self.active_fn(logit.float()) # calculate distogram
                    prob = prob.squeeze(0).permute(1,2,0).cpu().detach().numpy()
                    prob_s[i_logit][sub_idx_2d] = prob_s[i_logit][sub_idx_2d] + prob
                del logit_s, node
        #
        # combine all crops
        for i in range(4):
            prob_s[i] = prob_s[i] / count_2d[:,:,None]
        prob_in = np.concatenate(prob_s, axis=-1)
        node_s = node_s / count_1d[:, None]
        #
        # Do iterative refinement using SE(3)-Transformers
        # clear cache memory
        torch.cuda.empty_cache()
        #
        node_s = torch.tensor(node_s).to(self.device).unsqueeze(0)
        seq = msa[:,0].to(self.device)
        idx_pdb = idx_pdb.to(self.device)
        prob_in = torch.tensor(prob_in).to(self.device).unsqueeze(0)
        prob_s = prob_in
        with torch.cuda.amp.autocast():
            xyz, lddt = self.model(node_s, seq, idx_pdb, prob_s=prob_in, refine_only=True)
        return xyz, lddt, prob_s
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
    def for_fold(self, prob_s, xyz):
        # run TRFold
        prob_trF = list()
        for prob in prob_s:
            prob = torch.tensor(prob).permute(2,0,1).to(self.device)
            prob += 1e-8
            prob = prob / torch.sum(prob, dim=0)[None]
            prob_trF.append(prob)
        xyz = xyz[0, :, 1]
        TRF = TRFold(prob_trF, fold_params)
        xyz = TRF.fold(xyz, batch=15, lr=0.1, nsteps=200)
        xyz = xyz.detach().cpu().numpy()
        # add O and Cb
        N = xyz[:,0,:]
        CA = xyz[:,1,:]
        C = xyz[:,2,:]
        O = self.extend(np.roll(N, -1, axis=0), CA, C, 1.231, 2.108, -3.142)
        xyz = np.concatenate((xyz, O[:,None,:]), axis=1)
        return xyz
    def get_model_result(self, msa, xyz_t, t1d, t0d, window=150, shift=75):
        B, N, L = msa.shape
        # msa = torch.tensor(msa).long()
        xyz_t = xyz_t.float()
        t1d = t1d.float()
        t0d = t0d.float()
        t2d = xyz_to_t2d(xyz_t, t0d)
       
        # do cropped prediction if protein is too big
        if L > window*2:
            xyz, lddt, prob_s = self.for_cropped(msa, t0d, t1d, t2d, window, shift)
        else:
            xyz, lddt, prob_s = self.for_single(msa, t0d, t1d, t2d)
        return xyz, lddt, prob_s

    def extend(self, a,b,c, L,A,D):
        '''
        input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
        output: 4th coord
        '''
        N = lambda x: x/np.sqrt(np.square(x).sum(-1,keepdims=True) + 1e-8)
        bc = N(b-c)
        n = N(np.cross(b-a, bc))
        m = [bc,np.cross(n,bc),n]
        d = [L*np.cos(A), L*np.sin(A)*np.cos(D), -L*np.sin(A)*np.sin(D)]
        return c + sum([m*d for m,d in zip(m,d)])

if __name__ == "__main__":
    train = Train(use_cpu=True)
    # train.train("./generate_feat/train_data.pickle")
    # train.train_with_mask("./generate_feat/train_data.pickle")
    train.train_with_mask_v2("./generate_feat/train_data.pickle")
    # pred.predict(args.a3m_fn, args.out_prefix, None, args.atab)

