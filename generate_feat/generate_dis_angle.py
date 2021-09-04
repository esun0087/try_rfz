from re import X
import sys
import codecs
import os
import numpy as np
from multiprocessing import Pool
from Bio import PDB
import math
import torch
"""
CA x y z
"""

def get_pdbCA3(input_file):
    pdb_name = input_file.split('/')[-1].split(".")[0]
    ncaccb_path = os.path.join(out_base_dir, pdb_name, pdb_name + ".ncaccb.npy")
    if not os.path.exists(ncaccb_path):
        return None, None
    coords = np.load(ncaccb_path)
    return pdb_name, coords

def get_dis_class(v):
    segs = np.arange(2.5, 20.5, 0.5)
    for i,x in enumerate(segs):
        if i == 0 and v < x:
            return i
        if v < x and v > segs[i-1]:
            return i
    return len(segs)
def get_omega_class(v):
    segs = np.arange(0, math.pi * 2, 0.2)
    for i,x in enumerate(segs):
        if i == 0 and v < x:
            return i
        if v < x and v > segs[i-1]:
            return i
    return len(segs)

def get_phipsi_class(v):
    segs = np.arange(0, math.pi * 2, 0.1)
    for i,x in enumerate(segs):
        if i == 0 and v < x:
            return i
        if v < x and v > segs[i-1]:
            return i
    return len(segs)

# ============================================================
def get_pair_dist(a, b):
    """calculate pair distances between two sets of points
    
    Parameters
    ----------
    a,b : pytorch tensors of shape [batch,nres,3]
          store Cartesian coordinates of two sets of atoms
    Returns
    -------
    dist : pytorch tensor of shape [batch,nres,nres]
           stores paitwise distances between atoms in a and b
    """
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    dist = torch.cdist(a, b, p=2)
    return dist.numpy()

# ============================================================
def get_ang(a, b, c):
    """calculate planar angles for all consecutive triples (a[i],b[i],c[i])
    from Cartesian coordinates of three sets of atoms a,b,c 

    Parameters
    ----------
    a,b,c : pytorch tensors of shape [batch,nres,3]
            store Cartesian coordinates of three sets of atoms
    Returns
    -------
    ang : pytorch tensor of shape [batch,nres]
          stores resulting planar angles
    """
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    c = torch.from_numpy(c)
    v = a - b
    w = c - b
    v /= torch.norm(v, dim=-1, keepdim=True)
    w /= torch.norm(w, dim=-1, keepdim=True)
    vw = torch.sum(v*w, dim=-1)

    return torch.acos(vw).numpy()

# ============================================================
def get_dih(a, b, c, d):
    """calculate dihedral angles for all consecutive quadruples (a[i],b[i],c[i],d[i])
    given Cartesian coordinates of four sets of atoms a,b,c,d

    Parameters
    ----------
    a,b,c,d : pytorch tensors of shape [batch,nres,3]
              store Cartesian coordinates of four sets of atoms
    Returns
    -------
    dih : pytorch tensor of shape [batch,nres]
          stores resulting dihedrals
    """
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    c = torch.from_numpy(c)
    d = torch.from_numpy(d)
    b0 = a - b
    b1 = c - b
    b2 = d - c

    b1 /= torch.norm(b1, dim=-1, keepdim=True)

    v = b0 - torch.sum(b0*b1, dim=-1, keepdim=True)*b1
    w = b2 - torch.sum(b2*b1, dim=-1, keepdim=True)*b1

    x = torch.sum(v*w, dim=-1)
    y = torch.sum(torch.cross(b1,v,dim=-1)*w, dim=-1)

    return torch.atan2(y, x).numpy()


def process(out_base_dir, pdb):
    pdb_name = pdb.split('/')[-1].split(".")[0]
    out_dir = os.path.join(out_base_dir, pdb_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_path = os.path.join(out_dir, pdb_name + ".dis_angle")
    pdb_name, xyz = get_pdbCA3(pdb.strip())
    xyz = xyz.reshape(-1, 4, 3)
    if pdb_name is None:
        return
    N  = xyz[:,0]
    Ca = xyz[:,1]
    C  = xyz[:,2]
    Cb = xyz[:,3]
    L = xyz.shape[0]

    # 6d coordinates order: (dist,omega,theta,phi)
    c6d = np.full((L, L, 4), np.nan)

    dist = get_pair_dist(Ca,Ca)
    dist[np.isnan(dist)] = 999.9
    c6d[...,0] = dist + 999.9*np.eye(L)[None,...]
    i,j = np.where(c6d[...,0]<20)

    c6d[i,j,0] = c6d[i,j,0] - 2.5
    c6d[...,0][c6d[...,0] < 0] = 0
    c6d[i,j,0] = c6d[i,j,0] // 0.5
    c6d[...,0][c6d[...,0]>=36] = 36
    c6d[i,j,1] = get_dih(Ca[i], Cb[i], Cb[j], Ca[j]) + math.pi
    c6d[i,j,2] = get_dih(N[i], Ca[i], Cb[i], Cb[j]) + math.pi
    c6d[i,j,3] = get_ang(Ca[i], Cb[i], Cb[j]) + math.pi

    one_hot_phi = 36
    step1 = math.pi * 2 / one_hot_phi
    c6d[i,j,1] = c6d[i,j,1] // step1
    c6d[...,1][c6d[...,1]>=one_hot_phi] = one_hot_phi
    c6d[i,j,2] = c6d[i,j,2] // step1
    c6d[...,2][c6d[...,2]>=one_hot_phi] = one_hot_phi

    one_hot_omega = 16
    step2 = math.pi * 2 / one_hot_omega
    c6d[i,j,3] = c6d[i,j,3] // step2
    c6d[...,3][c6d[...,3]>=one_hot_omega] = one_hot_omega

    mask = np.zeros((L,L))
    mask[i,j] = 1.0
    # c6d[...,:1] = c6d[...,:1]*mask / 20
    # c6d[...,1:] = c6d[...,1:]*mask / (math.pi * 2)
    try:
        print ("save is", pdb, out_path)
        c6d = c6d.astype(np.int)
        np.save(out_path, c6d)
        out_path = os.path.join(out_dir, pdb_name + ".mask")
        np.save(out_path, mask)
    except Exception as e:
        print(e)
if __name__ == '__main__':
    out_base_dir = sys.argv[1]
    p = Pool(int(sys.argv[2]))
    for line in open("train-pdb.list"):
        p.apply_async(process, (out_base_dir, line.strip()))
    p.close()
    p.join()
