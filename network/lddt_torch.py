# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""lDDT protein distance score."""
# import numpy as jnp
import torch as jnp
from torch.nn.modules.loss import MSELoss

def lddt(predicted_points,
                 true_points,
                 cutoff=20.,
                 per_residue=False):
    """Measure (approximate) lDDT for a batch of coordinates.

    lDDT reference:
    Mariani, V., Biasini, M., Barbato, A. & Schwede, T. lDDT: A local
    superposition-free score for comparing protein structures and models using
    distance difference tests. Bioinformatics 29, 2722–2728 (2013).

    lDDT is a measure of the difference between the true distance matrix and the
    distance matrix of the predicted points.    The difference is computed only on
    points closer than cutoff *in the true structure*.

    This function does not compute the exact lDDT value that the original paper
    describes because it does not include terms for physical feasibility
    (e.g. bond length violations). Therefore this is only an approximate
    lDDT score.

    Args:
        predicted_points: (batch, length, 3) array of predicted 3D points
        true_points: (batch, length, 3) array of true 3D points
        true_points_mask: (batch, length, 1) binary-valued float array.    This mask
            should be 1 for points that exist in the true points.
        cutoff: Maximum distance for a pair of points to be included
        per_residue: If true, return score for each residue.    Note that the overall
            lDDT is not exactly the mean of the per_residue lDDT's because some
            residues have more contacts than others.

    Returns:
        An (approximate, see above) lDDT score in the range 0-1.
    """

    # Compute true and predicted distance matrices.
    dmat_true = jnp.sqrt(1e-10 + jnp.sum((true_points[:, :, None] - true_points[:, None, :])**2, axis=-1))

    dmat_predicted = jnp.sqrt(1e-10 + jnp.sum(
            (predicted_points[:, :, None] -
             predicted_points[:, None, :])**2, axis=-1))

    dists_to_score = (
            (dmat_true < cutoff).float() * \
            (1. - jnp.eye(dmat_true.shape[1]))    # Exclude self-interaction.
    )

    # Shift unscored distances to be far away.
    dist_l1 = jnp.abs(dmat_true - dmat_predicted)


    # True lDDT uses a number of fixed bins.
    # We ignore the physical plausibility correction to lDDT, though.
    score = 0.25 * ((dist_l1 < 0.5).float()   +
                                    (dist_l1 < 1.0).float()   +
                                    (dist_l1 < 2.0).float()   +
                                    (dist_l1 < 4.0).float()  )
    # Normalize over the appropriate axes.
    reduce_axes = (-1,) if per_residue else (-2, -1)
    norm = 1. / (1e-10 + jnp.sum(dists_to_score, axis=reduce_axes))
    score = norm * (1e-10 + jnp.sum(dists_to_score * score, axis=reduce_axes))

    return score
if __name__ == '__main__':

    predicted_points = jnp.randn(3, 5, 3, requires_grad=True)
    true_points = jnp.randn(3, 5, 3)
    # np.savetxt("p.txt", predicted_points)
    # np.savetxt("t.txt", true_points)
    # pred = np.load("test_lddt_predict.npy")
    # trues = np.load("test_lddt_true.npy")
    # pred = jnp.from_numpy(pred)
    # trues = jnp.from_numpy(trues)
    c = lddt(predicted_points, true_points)
    c = jnp.sum(c)
    c.backward()