"""DGM-Net loss functions.

Lovász-Softmax + OHEM cross-entropy for segmentation, plus the geometric
supervision terms (CCF regression, curvature, TV smoothness, boundary BCE).

Note: instantiate and move to your device, e.g.

    criterion = DGMNetLoss_Pro(ohem_thresh=0.9, ignore_index=255).to(device)

(the internal CE class weights / BCE pos_weight are registered as buffers,
so `.to(device)` handles device placement).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# Lovász-Softmax core ops
# ==========================================
def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes='present'):
    if probas.numel() == 0: return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if (classes == 'present' and fg.sum() == 0): continue
        class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return sum(losses) / len(losses)


def lovasz_softmax(probas, labels, ignore=255):
    probas = F.softmax(probas, dim=1)
    vprobas = probas.contiguous().view(-1, probas.size(1))
    vlabels = labels.contiguous().view(-1)
    valid = (vlabels != ignore)
    vprobas = vprobas[valid]
    vlabels = vlabels[valid]
    return lovasz_softmax_flat(vprobas, vlabels, classes='present')


# ==========================================
# Main loss
# ==========================================
class DGMNetLoss_Pro(nn.Module):
    def __init__(self, ohem_thresh=0.85, ignore_index=255, class_weights=None):
        super(DGMNetLoss_Pro, self).__init__()
        self.ignore_index = ignore_index
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights).float()
            self.ce = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=ignore_index, reduction='none')
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

        self.mse = nn.MSELoss(reduction='none')
        self.bce_edge = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]), reduction='none')
        self.thresh = ohem_thresh

    def total_variation_loss(self, img):
        b, c, h, w = img.size()
        tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
        return (tv_h + tv_w) / (b * c * h * w)

    def forward(self, pred_c, target_c, pred_dgm, target_dgm, pred_aux=None, geo_weight=2.0):
        pixel_losses = self.ce(pred_c, target_c)
        valid_mask = target_c != self.ignore_index

        # 1. OHEM cross-entropy (keep the hardest `thresh` fraction of pixels)
        flat_losses = pixel_losses.contiguous().view(-1)
        flat_mask = valid_mask.contiguous().view(-1)
        valid_flat_losses = flat_losses[flat_mask]
        if len(valid_flat_losses) > 0:
            num_kept = max(1, int(len(valid_flat_losses) * self.thresh))
            loss_ce_ohem = torch.topk(valid_flat_losses, k=num_kept)[0].mean()
        else:
            loss_ce_ohem = torch.tensor(0.0, device=pred_c.device)

        # 2. Lovász loss (direct IoU surrogate)
        loss_lovasz = lovasz_softmax(pred_c, target_c, ignore=self.ignore_index)
        loss_seg_global = loss_ce_ohem + 0.8 * loss_lovasz

        # 3. Boundary-weighted CE + geometric regression
        edge_weight = target_dgm[:, 3, :, :]
        loss_seg_boundary = (pixel_losses * edge_weight * valid_mask.float()).sum() / (edge_weight.sum() + 1e-6)

        valid_mask_float = valid_mask.unsqueeze(1).float()
        loss_vg_core = self.mse(pred_dgm[:, 0:3], target_dgm[:, 0:3])
        loss_curvature = self.mse(pred_dgm[:, 4:5], target_dgm[:, 4:5])

        loss_vg_total = (loss_vg_core * valid_mask_float).sum() / (valid_mask_float.sum() * 3 + 1e-6)
        loss_vg_total += (loss_curvature * valid_mask_float).sum() / (valid_mask_float.sum() + 1e-6)

        tv_loss = self.total_variation_loss(pred_dgm[:, 1:3])
        loss_vg = loss_vg_total + 0.5 * tv_loss

        loss_d = self.bce_edge(pred_dgm[:, 3:4], target_dgm[:, 3:4])
        loss_d = (loss_d * valid_mask_float).sum() / (valid_mask_float.sum() + 1e-6)

        loss_seg_total = loss_seg_global + 0.1 * loss_seg_boundary

        # 4. Auxiliary loss
        loss_aux = 0.0
        if pred_aux is not None:
            aux_pixel_losses = self.ce(pred_aux, target_c)
            loss_aux = (aux_pixel_losses * valid_mask.float()).sum() / (valid_mask.float().sum() + 1e-6)

        total = loss_seg_total + geo_weight * loss_vg + geo_weight * loss_d + 0.4 * loss_aux
        return total, loss_seg_total, loss_vg, loss_d
