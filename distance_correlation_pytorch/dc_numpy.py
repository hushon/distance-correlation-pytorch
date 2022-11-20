"""
NumPy implementation of Distance Correlation (DC) and Partial Distance Correlation (PDC)

Székely, Gábor J., Maria L. Rizzo, and Nail K. Bakirov. "Measuring and testing dependence by correlation of distances." The annals of statistics 35.6 (2007): 2769-2794.
Székely, Gábor J., and Maria L. Rizzo. "Partial distance correlation with methods for dissimilarities." The Annals of Statistics 42.6 (2014): 2382-2412.

https://stats.stackexchange.com/questions/183572/understanding-distance-correlation-computations
"""

import numpy as np
from numpy import ndarray


def distance_matrix(latent: ndarray) -> ndarray:
    """
    compute the double-centered matrix
    pp.6 in https://arxiv.org/pdf/1902.03291.pdf
    and pp.6 in https://arxiv.org/pdf/1410.1503.pdf
    """
    n = latent.shape[0]
    matrix_a = np.linalg.norm(latent[None,:,:] - latent[:,None,:], axis=-1)
    matrix_A = matrix_a - np.sum(matrix_a, axis=0, keepdims=True)/(n-2) - np.sum(matrix_a, axis=1, keepdims=True)/(n-2) + np.sum(matrix_a)/((n-1)*(n-2))
    np.fill_diagonal(matrix_A, 0.0)
    return matrix_A


def bracket_op(matrix_A: ndarray, matrix_B: ndarray) -> ndarray:
    r"""
    Covariance of signal A and B.
    :math:`\frac{1}{n(n-3)}` <A, B>
    """
    n = matrix_A.shape[0]
    return np.sum(matrix_A * matrix_B)/(n*(n-3))


def orthogonalize(matrix_A: ndarray, matrix_C: ndarray) -> ndarray:
    """
    A and B are 2D matrix
    Get orthogonal component of A to C
    return A - proj{C}{A}
    """
    orthogonal_projection = bracket_op(matrix_A, matrix_C) / bracket_op(matrix_C, matrix_C) * matrix_C
    result = matrix_A - orthogonal_projection
    return result


def correlation_sq(matrix_A: ndarray, matrix_B: ndarray) -> ndarray:
    """
    Pearson correlation coefficient
    """
    Gamma_XY = bracket_op(matrix_A, matrix_B)
    Gamma_XX = bracket_op(matrix_A, matrix_A)
    Gamma_YY = bracket_op(matrix_B, matrix_B)
    r_sq = Gamma_XY/(np.sqrt(Gamma_XX * Gamma_YY) + 1e-9)
    return r_sq


def squared_distance_correlation(feat_a: ndarray, feat_b: ndarray, cond_a: ndarray = None, cond_b: ndarray = None) -> ndarray:
    n = feat_a.shape[0]

    mat_A = distance_matrix(feat_a)
    mat_B = distance_matrix(feat_b)

    if cond_a is not None:
        mat_Acond = distance_matrix(cond_a)
        mat_A = orthogonalize(mat_A, mat_Acond)
    if cond_b is not None:
        mat_Bcond = distance_matrix(cond_b)
        mat_B = orthogonalize(mat_B, mat_Bcond)

    sq_dcov_ab = np.dot(mat_A.view(-1), mat_B.view(-1)) / (n*(n-3))
    sq_dcov_aa = np.dot(mat_A.view(-1), mat_A.view(-1)) / (n*(n-3))
    sq_dcov_bb = np.dot(mat_B.view(-1), mat_B.view(-1)) / (n*(n-3))

    sq_dcor_ab = sq_dcov_ab / np.sqrt(sq_dcov_aa * sq_dcov_bb)
    return sq_dcor_ab