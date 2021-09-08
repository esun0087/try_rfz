#!/usr/bin/python

import numpy as np
import torch
# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D2(A, B):
    A = A.detach()
    B = B.detach()
    A = A.permute(0, 2, 1)
    B = B.permute(0, 2, 1)
    A = A.numpy()
    B = B.numpy()
    R,t = rigid_transform_3D(A[0], B[0])
    R = torch.from_numpy(R).permute(1,0)
    t = torch.from_numpy(t).permute(1,0)
    return R, t

def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

if __name__ == '__main__':
    pointa = torch.randn(1, 100, 3)
    pointb = torch.randn(1, 100, 3)
    mse_loss = torch.nn.MSELoss()
    print(mse_loss(pointa, pointb))
    R, t = rigid_transform_3D2(pointa, pointb)
    c = torch.matmul(pointa, R) + t
    print(mse_loss(c, pointb))