import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    
    a = np.array(a)
    b = np.array(b)
    y = np.array(y)

    # compute Euclidean distance
    d = np.linalg.norm(a - b, axis=-1)

    # compute loss
    loss = y * (d ** 2) + (1 - y) * (np.maximum(0, margin - d) ** 2)

    # reduction
    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        return loss