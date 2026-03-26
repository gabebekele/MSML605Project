import numpy as np


def cosine_similarity(a, b, eps=1e-10):
    dot = np.sum(a * b, axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    return dot / (a_norm * b_norm + eps)


def euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=1)
