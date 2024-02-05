import numpy as np
import scipy.spatial.distance as dist


def Cosine(v1: list, v2: list):

    num = float(np.dot(v1, v2))
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

def Manhattan_distance(v1: np.ndarray, v2: np.ndarray):

    return np.linalg.norm(v1-v2, ord=1)


def Manhattan_distance_2(v1: list, v2: list):

    return sum(map(lambda i, j: abs(i-j), v1, v2))


def Euclidean_distance(v1: np.ndarray, v2: np.ndarray):

    return np.linalg.norm(v1 - v2)


def Hamming_distance(v1: list, v2: list):

    count = 0
    for i, j in zip(v1, v2):
        if i != j:
            count += 1
    return count


def Hamming_distance_2(v1: np.ndarray, v2: np.ndarray):

    smstr = np.nonzero(v1 - v2)
    sm = np.shape(smstr[0])[0]
    return sm


def jaccard_distance(v1: list, v2: list):

    int, union = 0, 0
    for i, j in zip(v1, v2):
        if i or j:
            union += 1
            if i and j:
                int += 1
    return 1 - (int / union)


def jaccard_distance_2(v1: np.ndarray, v2: np.ndarray):

    matv = np.mat([v1, v2])
    ds = dist.pdist(matv, 'jaccard')
    return ds[0]
