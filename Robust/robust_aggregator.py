# Put any functions you write for Question 1 here. 
import numpy as np

# For question 2, it would be helpful to define a function that returns the set of pruned gradients
def robust_aggregator(gradients):
    # Calculate covariance matrix
    cov_matrix = np.cov(gradients)
    # Calculate eigenvalues and eigenvectors
    eig_values, eig_vectors = np.linalg.eig(cov_matrix)
    # Sort eigenvalues and eigenvectors in order of decreasing eigenvalues
    sorted_indices = np.argsort(eig_values)[::-1]
    sorted_eig_values = eig_values[sorted_indices]
    sorted_eig_vectors = eig_vectors[:, sorted_indices]
    
    # Run pruning procedure
    pruned_gradients = gradients
    return pruned_gradients
