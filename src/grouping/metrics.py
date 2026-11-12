from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import adjusted_rand_score


# Adjusted Rand Index (ARI)

def compute_ari(true_groups, predicted_groups):
    """Compare reconstructed groups with reference case IDs."""
    return float(adjusted_rand_score(
        true_groups, predicted_groups
    ))


# Test-to-train maximum similarity (MaxSim)

def compute_maxsim(similarity_matrix):
    """
    similarity_matrix: shape [N_test, N_train]

    Returns the maximum similarity to the training set
    for each test image.
    """
    sim = np.asarray(similarity_matrix, dtype=np.float64)

    if sim.ndim != 2 or 0 in sim.shape:
        raise ValueError("Expected a non-empty 2D similarity matrix.")

    return np.max(sim, axis=1)


# High-Similarity Rate (HSR)

def compute_hsr(similarity_matrix, threshold=0.95):
    """
    Fraction of test images whose maximum similarity
    to any training image reaches the threshold.
    """
    max_sim = compute_maxsim(similarity_matrix)
    return float(np.mean(max_sim >= threshold))



# Mutual Nearest Neighbor (Mutual-NN)

def compute_mutual_nn(similarity_matrix):
    """
    Identify mutual nearest-neighbor pairs between
    the test set and the training set.

    Returns a boolean mask with one entry per test image.
    """
    sim = np.asarray(similarity_matrix, dtype=np.float64)

    if sim.ndim != 2 or 0 in sim.shape:
        raise ValueError("Expected a non-empty 2D similarity matrix.")

    nearest_train = np.argmax(sim, axis=1)
    nearest_test = np.argmax(sim, axis=0)

    return nearest_test[nearest_train] == np.arange(sim.shape[0])


def compute_mutual_nn_rate(similarity_matrix):
    """Fraction of test images forming mutual NN pairs."""
    return float(np.mean(compute_mutual_nn(similarity_matrix)))


