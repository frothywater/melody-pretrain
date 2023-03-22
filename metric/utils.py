# coding:utf-8
"""utils.py
Include distance calculation for evaluation metrics
Copied from: https://github.com/RichardYang40148/mgeval
"""
import numpy as np
import sklearn
from scipy import integrate, stats
from sklearn.model_selection import LeaveOneOut


def c_dist(A, B, mode="None", normalize=0):
    c_dist = np.zeros(len(B))
    for i in range(0, len(B)):
        if mode == "None":
            c_dist[i] = np.linalg.norm(A - B[i])
        elif mode == "EMD":
            if normalize == 1:
                A_ = sklearn.preprocessing.normalize(A.reshape(1, -1), norm="l1")[0]
                B_ = sklearn.preprocessing.normalize(B[i].reshape(1, -1), norm="l1")[0]
            else:
                A_ = A.reshape(1, -1)[0]
                B_ = B[i].reshape(1, -1)[0]

            c_dist[i] = stats.wasserstein_distance(A_, B_)

        elif mode == "KL":
            if normalize == 1:
                A_ = sklearn.preprocessing.normalize(A.reshape(1, -1), norm="l1")[0]
                B_ = sklearn.preprocessing.normalize(B[i].reshape(1, -1), norm="l1")[0]
            else:
                A_ = A.reshape(1, -1)[0]
                B_ = B[i].reshape(1, -1)[0]

            B_[B_ == 0] = 0.00000001
            c_dist[i] = stats.entropy(A_, B_)
    return c_dist


def cross_valid(A: np.ndarray, B: np.ndarray):
    loo = LeaveOneOut()
    num_samples = len(A)
    loo.get_n_splits(np.arange(num_samples))
    result = np.zeros((num_samples, num_samples))
    for _, test_index in loo.split(np.arange(num_samples)):
        result[test_index[0]] = c_dist(A[test_index], B)
    return result.flatten()


def overlap_area(A, B):
    """Calculate overlap between the two PDF"""
    pdf_A = stats.gaussian_kde(A)
    pdf_B = stats.gaussian_kde(B)
    return integrate.quad(
        lambda x: min(pdf_A(x), pdf_B(x)), np.min((np.min(A), np.min(B))), np.max((np.max(A), np.max(B)))
    )[0]


def kl_dist(A, B, num_sample=1000):
    """Calculate KL distance between the two PDF"""
    pdf_A = stats.gaussian_kde(A)
    pdf_B = stats.gaussian_kde(B)
    sample_A = np.linspace(np.min(A), np.max(A), num_sample)
    sample_B = np.linspace(np.min(B), np.max(B), num_sample)
    return stats.entropy(pdf_A(sample_A), pdf_B(sample_B))
