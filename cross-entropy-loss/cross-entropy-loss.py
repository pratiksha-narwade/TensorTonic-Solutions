import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    
    # number of samples
    N = len(y_true)
    
    # small value to avoid log(0)
    epsilon = 1e-15
    
    # clip probabilities
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # select predicted probabilities for correct classes
    correct_probs = y_pred[np.arange(N), y_true]
    
    # compute loss
    loss = -np.mean(np.log(correct_probs))
    
    return loss