import torch
import numpy as np

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for i in range(100):
    # Set a seed for reproducibility
    #np.random.seed(11) #11=8 0.8, 42, 0.2 = 9
    rank1 = np.random.randint(1, 100)
    m1 = np.dot(np.random.rand(100, rank1), np.random.rand(rank1, 100))  # 100 samples, 50 features but rank 10
    rank2 = np.random.randint(1, 100)
    m2 = np.dot(np.random.rand(100, rank2), np.random.rand(rank2, 100))  # 100 samples, 50 features but rank 10

    random_matrix = m1@m2

    # Calculate covariance matrix
    cov_matrix = np.cov(random_matrix.T)

    # SVD decomposition
    U, S, Vh = np.linalg.svd(cov_matrix)

    # Calculate effective rank 
    threshold = 1e-3
    effective_rank = np.sum(S > threshold)

    # Print results

    print(f"Effective rank: {effective_rank}")

    print(f"min: {min(rank1, rank2)}")
