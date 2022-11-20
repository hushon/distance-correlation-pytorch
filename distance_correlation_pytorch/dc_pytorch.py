"""
Distance correlation (DC) for differentiable objective function
DC is computed per mini-batch
"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torchsort


def distance_matrix(feat: Tensor) -> Tensor:
    """
    feat.shape == (N, dim)
    output.shape == (N, N)
    """
    n = feat.size(0)
    matrix_a = torch.norm(feat.unsqueeze(0) - feat.unsqueeze(1), dim=-1) # distance matrix
    matrix_A = matrix_a - matrix_a.sum(0, keepdims=True)/(n-2) - matrix_a.sum(1, keepdims=True)/(n-2) + matrix_a.sum(matrix_a)/((n-1)*(n-2)) # U-centered matrix
    matrix_A.fill_diagonal_(0.0)
    return matrix_A


def orthogonalize(vec_A: Tensor, vec_C: Tensor) -> Tensor:
    """
    Do orthogonal decomosition of vec_A and
    get the component that is perpendicular to vec_C
    returns A - proj{C}{A}
    """
    orthogonal_proj = torch.dot(vec_A, vec_C) / torch.dot(vec_C, vec_C) * vec_C
    result = vec_A - orthogonal_proj
    return result


def spearman_correlation(val_A: Tensor, val_B: Tensor, regularization_strength=1.0) -> Tensor:
    val_A = torchsort.soft_rank(val_A, regularization_strength=regularization_strength)
    val_B = torchsort.soft_rank(val_B, regularization_strength=regularization_strength)
    val_A = val_A - val_A.mean()
    val_A = val_A / val_A.norm()
    val_B = val_B - val_B.mean()
    val_B = val_B / val_B.norm()
    return torch.dot(val_A, val_B)


def squared_distance_correlation(feat_a: Tensor, feat_b: Tensor, cond_a: Tensor = None, cond_b: Tensor = None) -> Tensor:
    assert feat_a.size(0) == feat_b.size(0)
    n = feat_a.size(0)

    vec_A = distance_matrix(feat_a).view(-1)
    vec_B = distance_matrix(feat_b).view(-1)

    # project to the orthogonal subspace of conditioning variables
    if cond_a is not None:
        vec_Acond = distance_matrix(cond_a).view(-1)
        vec_A = orthogonalize(vec_A, vec_Acond)
    if cond_b is not None:
        vec_Bcond = distance_matrix(cond_b).view(-1)
        vec_B = orthogonalize(vec_B, vec_Bcond)

    sq_dcov_ab = torch.dot(vec_A, vec_B).div(n*(n-3)) # squared distance covariances
    sq_dcov_aa = torch.dot(vec_A, vec_A).div(n*(n-3))
    sq_dcov_bb = torch.dot(vec_B, vec_B).div(n*(n-3))

    sq_dcor_ab = sq_dcov_ab / torch.sqrt(sq_dcov_aa * sq_dcov_bb).clamp(1e-9) # squared distance correlation
    return sq_dcor_ab


class DistanceCorrelationLoss(nn.Module):
    """
    computes squared distance correlation using unbiased estimation formula.
    when cond_a or cond_b is not none, computes squared partial distance correlation.
    """
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def get_feature_closure(layer: nn.Module) -> callable:
        feature = [None]
        def forward_hook(module, input, output):
            feature[0] = output
        handle = layer.register_forward_hook(forward_hook)
        def get_feature():
            return feature[0]
        return get_feature, handle

    def forward(self, feat_a: Tensor, feat_b: Tensor, cond_a: Tensor = None, cond_b: Tensor = None) -> Tensor:
        return squared_distance_correlation(feat_a, feat_b, cond_a, cond_b)
