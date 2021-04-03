import scipy.spatial

def cdist(A, B, **kwargs):
    return scipy.spatial.distance.cdist(A, B, **kwargs)