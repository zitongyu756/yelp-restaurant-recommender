import numpy as np

def kmeans(X: np.ndarray, k: int, max_iters: int = 100, tol: float = 1e-4, seed: int = 42):
    """
    K-means clustering implemented from scratch using NumPy.
    
    Args:
        X: (N, D) array of data points
        k: Number of clusters
        max_iters: Maximum number of iterations
        tol: Tolerance for convergence
        seed: Random seed for initialization
        
    Returns:
        centroids: (k, D) array of cluster centers
        labels: (N,) array of cluster assignments
    """
    np.random.seed(seed)
    N, D = X.shape
    
    # Initialize centroids randomly from the data points
    initial_indices = np.random.choice(N, k, replace=False)
    centroids = X[initial_indices].copy()
    
    labels = np.zeros(N, dtype=int)
    
    for _ in range(max_iters):
        # Compute squared Euclidean distances to avoid sqrt overhead
        # X^2 - 2X*C + C^2
        X_sq = np.sum(X**2, axis=1, keepdims=True)
        C_sq = np.sum(centroids**2, axis=1)
        distances_sq = X_sq - 2 * np.dot(X, centroids.T) + C_sq
        
        # Assign each point to the closest centroid
        new_labels = np.argmin(distances_sq, axis=1)
        
        # Check for convergence
        if np.array_equal(labels, new_labels):
            break
            
        labels = new_labels
        
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = np.mean(cluster_points, axis=0)
            else:
                # Handle empty cluster
                new_centroids[j] = X[np.random.choice(N)]
                
        # Check tolerance
        shift = np.linalg.norm(new_centroids - centroids)
        if shift < tol:
            break
            
        centroids = new_centroids
        
    return centroids, labels
