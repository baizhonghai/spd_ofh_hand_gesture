import numpy as np
from scipy.linalg import logm


def general_mpower(X, Y):
    # Diagonalize the matrix
    d, V = np.linalg.eig(X)

    # Element-wise exponentiation of d
    d_power = np.diag(d ** Y)

    # Element-wise multiplication of V and d_power
    V_d_power = V * d_power

    # Transpose of V_d_power
    V_d_power_transpose = V_d_power.conj().T

    # Matrix multiplication (V_d_power_transpose) * V
    Z = np.dot(V_d_power_transpose, V)
    return Z


def compute_ldv(X, Y):
    X_m0d5 = general_mpower(X, -0.5)
    temp_gradient, temp_dis = log_euclidean_metric(X, Y)
    current_gradient = np.dot(X_m0d5, np.dot(temp_gradient, X_m0d5))
    current_dis = temp_dis
    if current_dis < 1e-10 or np.linalg.norm(current_gradient, 'fro') < 1e-10:
        temp_ldv = np.zeros_like(current_gradient)
    else:
        temp_ldv = (current_gradient / np.linalg.norm(current_gradient, 'fro')) * current_dis
    rie_ldv = map2IDS_vectorize(temp_ldv, 0)
    return rie_ldv

def log_euclidean_metric(X, Y):
    temp_norm = logm(X) - logm(Y)
    temp_gradient = np.dot(np.linalg.inv(X), temp_norm)
    temp_gradient = 2 * np.dot(X, (temp_gradient + temp_gradient.T)) * X
    temp_dis = np.linalg.norm(temp_norm, 'fro')
    return temp_gradient, temp_dis

def map2IDS_vectorize(inMat, map2IDS):
    if map2IDS == 1:
        inMat = logm(inMat)
    offDiagonals = np.tril(inMat, -1) * np.sqrt(2)
    diagonals = np.diag(np.diag(inMat))
    vecInMat = diagonals + offDiagonals
    vecInds = np.tril(np.ones_like(inMat))
    map2ITS = vecInMat.ravel()
    vecInds = vecInds.ravel()
    y = map2ITS[vecInds == 1]
    return y

if __name__=="__main__":
    np.random.seed(42)
    # Generate random feature matrices
    #feature_matrix1 = np.random.rand(3, 100)
    #feature_matrix2 = np.random.rand(3, 100)
    feature_matrix1 = np.array([[0.1122, 0.345, 0.432],
                                [0.2455, 0.5849, 0.6738],
                                [0.321, 0.753, 0.9287]])

    feature_matrix2 = np.array([[0.9817, 0.753, 0.34121],
                                [0.678, 0.1589, 0.245],
                                [0.4432, 0.345, 0.1122]])
    # Compute covariance matrices
    X = np.cov(feature_matrix1)
    Y = np.cov(feature_matrix2)

    # LDV while using LEM
    ldv_L = compute_ldv(X, Y)

    print("LDV with LEM:\n", ldv_L)
