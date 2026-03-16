import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    
    # avoid log(0) and division by 0
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    
    return np.sum(p * np.log(p / q))