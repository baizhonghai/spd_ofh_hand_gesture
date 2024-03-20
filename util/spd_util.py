import numpy as np
from scipy.linalg import logm

def map2IDS_vectorize(inMat):
    inMat = np.real(logm(inMat))
    offDiagonals = np.tril(inMat, -1) * np.sqrt(2)
    diagonals = np.diag(np.diag(inMat))
    vecInMat = diagonals + offDiagonals
    vecInds = np.tril(np.ones_like(inMat))
    map2ITS = vecInMat.flatten()
    vecInds = vecInds.flatten()
    y = map2ITS[vecInds == 1]
    return y