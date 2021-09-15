import torch
import rigid_transform_3D
class Loss:
    def __init__(self, device) -> None:
        self.device = device
        pass

    def cross_loss_mask(self, pred_, true_, mask):
        pred_ = pred_.reshape(-1, pred_.shape[-1])
        true_ = torch.flatten(true_)
        mask = torch.flatten(mask).float()
        cross_func = torch.nn.CrossEntropyLoss(reduction='none')
        loss = cross_func(pred_, true_)
        loss = mask * loss
        result = torch.mean(loss)
        return result

    def coords_loss_rotate(self,pred_, true_):
        def get_r_t(pred_, true_):
            true_ca = true_.view(-1, 3, 3)[:,1,:]
            pred_ca = pred_.view(-1, 3, 3)[:,1,:]
            select_atoms = torch.where(~torch.isnan(true_ca))
            true_coords = true_ca[select_atoms].view(-1, 3)
            pred_coords = pred_ca[select_atoms].view(-1, 3)
            R, t = rigid_transform_3D.rigid_transform_3D2(pred_coords, true_coords)
            return R, t
        losses = 0
        for pred_true, cur_true in zip(pred_, true_):
            mse_loss = torch.nn.MSELoss()
            cur_mask = ~torch.isnan(true_)
            R, t = get_r_t(pred_true, cur_true)
            pred_true = cur_mask * pred_true
            pred_true.masked_fill_(cur_mask, 0)
            cur_true = cur_true.masked_fill(cur_mask, 0)
            pred_rotate = torch.matmul(pred_, R) + t
            c = mse_loss(pred_rotate, cur_true)
            c = torch.sqrt(c)
            losses = losses + c
        return losses

    def dis_mse_whole_atom(self, predicted_points, true_points):
        """
        compute whole matrix loss
        """
        # Compute true and predicted distance matrices.
        dmat_true = torch.sqrt(1e-10 + torch.sum((true_points[:, :, None] - true_points[:, None, :])**2, axis=-1))

        dmat_predicted = torch.sqrt(1e-10 + torch.sum(
                (predicted_points[:, :, None] -
                predicted_points[:, None, :])**2, axis=-1))
        mask = (~torch.isnan(dmat_true)).float() * (1 - torch.eye(true_points.shape[1]))
        
        dmat_true.masked_fill_(~mask.bool(), 0)
        dmat_predicted = mask * dmat_predicted
        dmat_true = mask * dmat_true
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(dmat_predicted, dmat_true)
        loss = torch.sqrt(loss)
        return loss

    def dis_mse_loss_ca(self, predicted_points, true_points):
        """
        is just like lddt
        """
        B = true_points.shape[0]
        # ca
        true_points = true_points.view(B, -1, 3, 3)[:,:,1]
        predicted_points = predicted_points.view(B, -1, 3, 3)[:,:,1]
        # Compute true and predicted distance matrices.
        dmat_true = torch.sqrt(1e-10 + torch.sum((true_points[:, :, None] - true_points[:, None, :])**2, axis=-1))

        dmat_predicted = torch.sqrt(1e-10 + torch.sum(
                (predicted_points[:, :, None] -
                predicted_points[:, None, :])**2, axis=-1))

        tmp = torch.abs(dmat_true - dmat_predicted)
        mask = (dmat_true < 4).float() * (1 - torch.eye(true_points.shape[1]))
        #score = 0.25 * ((tmp < 0.5).float() + (tmp < 1.0).float() + (tmp < 2.0).float() + (tmp < 4.0).float())

        dmat_true.masked_fill_(~mask.bool(), 0)
        mse_loss = torch.nn.MSELoss(reduction='none')
        loss = mse_loss(dmat_predicted, dmat_true)
        loss = mask * loss
        #loss = score * loss
        loss = torch.sqrt(torch.mean(loss))
        return loss