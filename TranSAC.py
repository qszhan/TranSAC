import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.interpolate import BSpline
from collections import Counter






def fda_paras(T, n_basis, order):
    knots = np.linspace(0, 1, n_basis + order)
    n_samples, n_features = T.shape
    grid_points = np.linspace(0, 1, n_features)
    T_Bspline_para = np.zeros((n_samples, n_basis))
    for i in range(n_samples):
        spl = BSpline(grid_points, T[i], order - 1)
        T_Bspline_para[i] = spl.__call__(knots[:-order])
    return T_Bspline_para


def pt_KDE(T, bandwidth):
    # Fit the kernel density estimator to T
    kde = KernelDensity(bandwidth=bandwidth).fit(T)
    # Evaluate the density at a set of query points T
    log_densities = kde.score_samples(T)
    true_probs = np.exp(log_densities)
    true_probs_normalized = true_probs / np.sum(true_probs)
    return true_probs_normalized


def entropy_ZT(p_zhat_given_t, pt, K):
    ht = -np.sum(pt * np.log2(pt), axis=0)
    predicted_z = np.argmax(p_zhat_given_t, axis=1)
    # Count occurrences of each class in predictions
    counts = Counter(predicted_z)
    # Calculate probability of each class in label space
    pz =  [counts.get(c, 0) / K for c in range(0, K)]
    pz = np.array(pz)
    hz = -np.sum(pz * np.log2(pz+1e-10), axis=0)
    return ht, hz


def TranSAC(tar_prob, tar_fea, n_basis, order):
    # dimension reduction using based on FDA
    tar_features_fda = fda_paras(np.array(tar_fea), n_basis, order)
    # obtain pt using the KDE
    pt = pt_KDE(tar_features_fda, bandwidth=0.01)
    p_z_given_t = np.array(tar_prob)
    K = len(p_z_given_t[0])
    ht, hz = entropy_ZT(p_z_given_t, pt, K)
    TranSAC_Plus = ht + hz
    TranSAC_Minus = ht - hz
    return TranSAC_Plus, TranSAC_Minus



if __name__ == "__main__":
    tar_num = 100
    src_class_num = 1000
    tar_repre_dim = 2048
    # generate and normalise the target representations
    tar_represent = np.random.rand(tar_num, tar_repre_dim)
    tar_represent_norm = tar_represent / np.linalg.norm(tar_represent, axis=1, keepdims=True)
    # generate target predictions over the source label space
    tar_predict = np.random.random((tar_num, src_class_num))
    tar_predict /= np.sum(tar_predict, axis=1, keepdims=True)
    TranSAC_Plus,  TranSAC_Minus = TranSAC(tar_predict, tar_represent_norm, n_basis=15, order=3)

