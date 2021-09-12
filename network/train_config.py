import numpy as np
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
        "performer_N_opts": {"nb_features": 16},
        "performer_L_opts": {"nb_features": 16}
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