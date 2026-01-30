import numpy as np

def sigmoid(x):
    """
    Numerically stable, vectorized sigmoid function.
    """
    x = np.array(x)  
    result = np.empty_like(x, dtype=float)

    positive_mask = x >= 0
    negative_mask = ~positive_mask

    result[positive_mask] = 1 / (1 + np.exp(-x[positive_mask]))

    exp_x = np.exp(x[negative_mask])
    result[negative_mask] = exp_x / (1 + exp_x)

    return result
