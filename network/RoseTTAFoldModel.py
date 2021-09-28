import torch
import torch.nn as nn
from Embeddings import MSA_emb, Pair_emb_wo_templ, Pair_emb_w_templ, Templ_emb
from Attention_module_w_str import IterativeFeatureExtractor
from DistancePredictor import DistanceNetwork
from Refine_module import Refine_module

class RoseTTAFoldModule(nn.Module):
    def __init__(self, n_module=4, n_module_str=4, n_layer=4,\
                 d_msa=64, d_pair=128, d_templ=64,\
                 n_head_msa=4, n_head_pair=8, n_head_templ=4,
                 d_hidden=64, r_ff=4, n_resblock=1, p_drop=0.1, 
                 performer_L_opts=None, performer_N_opts=None,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}, 
                 use_templ=False):
        super(RoseTTAFoldModule, self).__init__()
        self.use_templ = use_templ
        #
        self.msa_emb = MSA_emb(d_model=d_msa, p_drop=p_drop, max_len=5000)
        if use_templ:
            self.templ_emb = Templ_emb(d_templ=d_templ, n_att_head=n_head_templ, r_ff=r_ff, 
                                       performer_opts=performer_L_opts, p_drop=0.0)
            self.pair_emb = Pair_emb_w_templ(d_model=d_pair, d_templ=d_templ, p_drop=p_drop)
        else:
            self.pair_emb = Pair_emb_wo_templ(d_model=d_pair, p_drop=p_drop)
        #
        self.feat_extractor = IterativeFeatureExtractor(n_module=n_module,\
                                                        n_module_str=n_module_str,\
                                                        n_layer=n_layer,\
                                                        d_msa=d_msa, d_pair=d_pair, d_hidden=d_hidden,\
                                                        n_head_msa=n_head_msa, \
                                                        n_head_pair=n_head_pair,\
                                                        r_ff=r_ff, \
                                                        n_resblock=n_resblock,
                                                        p_drop=p_drop,
                                                        performer_N_opts=performer_N_opts,
                                                        performer_L_opts=performer_L_opts,
                                                        SE3_param=SE3_param)
        self.c6d_predictor = DistanceNetwork(d_pair, p_drop=p_drop)

    def forward(self, msa, seq, idx, t1d=None, t2d=None):
        B, N, L = msa.shape
        # Get embeddings
        msa = self.msa_emb(msa, idx)
        if self.use_templ:
            tmpl = self.templ_emb(t1d, t2d, idx)
            pair = self.pair_emb(seq, idx, tmpl)
        else:
            pair = self.pair_emb(seq, idx)
        #
        # Extract features
        seq1hot = torch.nn.functional.one_hot(seq, num_classes=21).float()
        msa, pair, xyz, lddt = self.feat_extractor(msa, pair, seq1hot, idx)

        # Predict 6D coords
        logits = self.c6d_predictor(pair)
        
        return logits, xyz, lddt.view(B, L)


class RoseTTAFoldModule_e2e(nn.Module):
    def __init__(self, n_module=4, n_module_str=4, n_module_ref=4, n_layer=4,\
                 d_msa=64, d_pair=128, d_templ=64,\
                 n_head_msa=4, n_head_pair=8, n_head_templ=4,
                 d_hidden=64, r_ff=4, n_resblock=1, p_drop=0.0, 
                 performer_L_opts=None, performer_N_opts=None,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}, 
                 REF_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}, 
                 use_templ=False):
        super(RoseTTAFoldModule_e2e, self).__init__()
        self.use_templ = use_templ
        #
        self.msa_emb = MSA_emb(d_model=d_msa, p_drop=p_drop, max_len=5000)
        if use_templ:
            self.templ_emb = Templ_emb(d_templ=d_templ, n_att_head=n_head_templ, r_ff=r_ff, 
                                       performer_opts=performer_L_opts, p_drop=0.0)
            self.pair_emb = Pair_emb_w_templ(d_model=d_pair, d_templ=d_templ, p_drop=p_drop)
        else:
            self.pair_emb = Pair_emb_wo_templ(d_model=d_pair, p_drop=p_drop)
        #
        self.feat_extractor = IterativeFeatureExtractor(n_module=n_module,\
                                                        n_module_str=n_module_str,\
                                                        n_layer=n_layer,\
                                                        d_msa=d_msa, d_pair=d_pair, d_hidden=d_hidden,\
                                                        n_head_msa=n_head_msa, \
                                                        n_head_pair=n_head_pair,\
                                                        r_ff=r_ff, \
                                                        n_resblock=n_resblock,
                                                        p_drop=p_drop,
                                                        performer_N_opts=performer_N_opts,
                                                        performer_L_opts=performer_L_opts,
                                                        SE3_param=SE3_param)
        self.c6d_predictor = DistanceNetwork(d_pair, p_drop=p_drop)
        #
        self.refine = Refine_module(n_module_ref, d_node=d_msa, d_pair=130,
                                    d_node_hidden=d_hidden, d_pair_hidden=d_hidden,
                                    SE3_param=REF_param, p_drop=p_drop)

    def get_msa_mask(self, feat, lens_info):
        B = feat.shape[0]
        mask_like = torch.ones_like(feat) # 不要的地方被设置为1
        for idx in range(B):
            mask_like[idx][:,:lens_info[idx]] = 0
        mask_like = mask_like.bool()
        return mask_like

    def forward(self, msa, seq, idx, t1d=None, t2d=None, prob_s=None, lens_info = None):
        seq1hot = torch.nn.functional.one_hot(seq, num_classes=21).float()
        B, N, L = msa.shape
        # Get embeddings
        msa = self.msa_emb(msa, idx) # idx 主要是为了添加位置信息, t1d是为了添加ij的匹配信息 
        msa_mask = self.get_msa_mask(msa, lens_info)
        msa.masked_fill_(msa_mask, 0)

        if self.use_templ:
            tmpl = self.templ_emb(t1d, t2d, idx, lens_info)
            pair = self.pair_emb(seq, idx, tmpl, lens_info) # 感觉是把序列embeding信息，idx一维位置信息，强行扩展，添加到了二维信息里.分为横向和纵向扩展
        else:
            pair = self.pair_emb(seq, idx)
        ret_msa = torch.empty((B, L, msa.shape[-1]))
        ret_pair = torch.empty_like(pair)
        ret_xyz = torch.empty((B, L, 3, 3))
        ret_lddt = torch.empty((B, L))

        # msa, pair, xyz, lddt = self.feat_extractor(msa, pair, seq1hot, idx)
        # print(msa.shape, pair.shape, xyz.shape, lddt.shape)
        #
        # Extract features
        for i in range(B):
            cur_l = lens_info[i]
            cur_msa = msa[i, :,:cur_l]
            cur_pair = pair[i, :cur_l, :cur_l]
            cur_seq1hot = seq1hot[i, :cur_l]
            cur_idx = idx[i, :cur_l]
            cur_msa, cur_pair, cur_seq1hot, cur_idx = cur_msa.unsqueeze(0), cur_pair.unsqueeze(0), cur_seq1hot.unsqueeze(0),cur_idx.unsqueeze(0)
            # print(cur_msa.shape, cur_pair.shape, cur_seq1hot.shape, cur_idx.shape)
            ret_msa[i, :cur_l], ret_pair[i, :cur_l, :cur_l], ret_xyz[i, :cur_l], ret_lddt[i, :cur_l] = self.feat_extractor(cur_msa, cur_pair, cur_seq1hot, cur_idx)

        # Predict 6D coords
        logits = self.c6d_predictor(ret_pair)
        prob_s = list()
        for l in logits:
            if torch.sum(torch.isnan(l)) > 0:
                print("logits nan")
            prob_s.append(nn.Softmax(dim=1)(l)) # (B, C, L, L)
        prob_s = torch.cat(prob_s, dim=1).permute(0,2,3,1)

        # ret_xyz = torch.empty(B, L, 3, 3)
        # ret_lddt = torch.empty(B, L, 1)
        # for i in range(B):
        #     cur_l = lens_info[i]
        #     cur_msa, cur_probs, cur_seq1hot, cur_idx = ret_msa[i, :cur_l].unsqueeze(0), prob_s[i, :cur_l, :cur_l].unsqueeze(0), \
        #         seq1hot[i,:cur_l].unsqueeze(0), idx[i, :cur_l].unsqueeze(0)
        #     ret_xyz[i, :cur_l], ret_lddt[i, :cur_l] = self.refine(cur_msa, cur_probs, cur_seq1hot, cur_idx)
        ret_xyz, ret_lddt = self.refine(ret_msa, prob_s, seq1hot, idx)
        if torch.sum(torch.isnan(ret_msa)) > 0:
            print("msa nan")
        if torch.sum(torch.isnan(ret_xyz)) > 0:
            print("ret_xyz nan")
        if torch.sum(torch.isnan(ret_lddt)) > 0:
            print("ret_lddt nan")
        return logits, ret_msa, ret_xyz, ret_lddt.view(B,L)
