import numpy as np
from scipy.linalg import inv

def random_dag(n, p, rng):
    """Return adjacency (upper‑triangular) for a random DAG with edge prob p."""
    A = (rng.random((n, n)) < p).astype(float)
    A = np.triu(A, k=1)   # ensure acyclic by using upper triangle
    return A

def sample_sem(A, n_samples, rng):
    """Linear‐Gaussian SEM: X = Aᵀ X + ε with ε~N(0, I)."""
    n = A.shape[0]
    # solve (I - Aᵀ) X = ε
    I_minus_AT = np.eye(n) - A.T
    eps = rng.standard_normal((n_samples, n))
    X = eps @ inv(I_minus_AT)
    return X