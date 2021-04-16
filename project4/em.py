"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    # Get model parameters
    n = X.shape[0]
    mu, var, pi = mixture
    K = mu.shape[0]

    # To store the normal matrix and log of posterior probs -> (p(j|u))
    f = np.zeros((n,K), dtype=np.float64)
    
    for i in range(n):
        # Get indexes of columns with ratings for each user
        cu_indices = X[i,:] > 0
        cu_size = np.sum(cu_indices)

        pre_exp = (-0.5 * cu_size) * np.log((2 * np.pi * var)) # log of pre-exponent for the i user gaussian distribution
        diff = X[i, cu_indices] - mu[:, cu_indices]   # Exponent term of the gaussian. Shape(K,|Cu|)
        norms = np.sum(diff**2, axis=1)  # shape (K,)

        f[i,:] = pre_exp - 0.5 * norms / var  # Shape (K,)

    f = f + np.log(pi + 1e-16)  # f(u,j) matrix
    
    # Log of normal term in p(j|u)
    sums = logsumexp(f, axis=1).reshape(-1,1) 
    posts = f - sums # Log of posterior probs matrix -> log(p(j|u))
    
    log_likelihood = np.sum(sums, axis=0).item()
    
    return np.exp(posts), log_likelihood



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    # Get model parameters
    n = X.shape[0]
    mu_new, _, _ = mixture
    K = mu_new.shape[0]
    
    pi_new = np.sum(post, axis=0) / n # New pi(j)
    delta = X.astype(bool).astype(int) #  Delta function: where X is non-zero
    
    denom = post.T @ delta # Shape (K,d). Hadamard product
    numer = post.T @ X # Shape (K,d). Hadamard product 
    updatable_indices = np.where(denom >= 1)
    mu_new[updatable_indices] = numer[updatable_indices] / denom[updatable_indices] # Update where denom >= 1
    
    denom_var = np.sum(post * np.sum(delta, axis=1).reshape(-1,1), axis=0) # Update variances. Shape: (K,)
    norms = np.zeros((n, K), dtype=np.float64)
    
    for i in range(n):
        cu_index = X[i,:] > 0 # Columns with ratings for each user
        diff = X[i, cu_index] - mu_new[:, cu_index] # Shape (K,|Cu|)
        norms[i,:] = np.sum(diff**2, axis=1) # Shape (K,)
    
    # New variances. if var(j) < 0.25, then var(j) = 0.25
    var_new = np.maximum(np.sum(post * norms, axis=0) / denom_var, min_variance)  
    
    return GaussianMixture(mu_new, var_new, pi_new)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_likelihood = None
    new_likelihood = None
    
    while old_likelihood is None or (new_likelihood - old_likelihood > 1e-6 * np.abs(new_likelihood)):
        old_likelihood = new_likelihood
        
        post, new_likelihood = estep(X, mixture) # E step
        mixture = mstep(X, post, mixture) # M step
            
    return mixture, post, new_likelihood


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    Y = X.copy() # Predictions
    mu, _, _ = mixture # Get model parameters
    
    post, _ = estep(X, mixture) # Get posteriors
    
    miss_index = np.where(X == 0) # Ratings to be filled
    Y[miss_index] = (post @ mu)[miss_index]
    
    return Y
