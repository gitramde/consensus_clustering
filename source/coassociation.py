import numpy as np
from config import parameters
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

def coassociation_matrix(n_samples, base_clustering):
    coassoc = np.zeros((n_samples, n_samples))
    for labels in base_clustering:
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                if labels[i] == labels[j]:
                    coassoc[i, j] += 1
                    coassoc[j, i] += 1
    coassoc /= parameters.n_base_clusterings # normalize to [0, 1]
    return coassoc

def hierarchial_clustering(coassoc_matrix, n_clusters):
    np.fill_diagonal(coassoc_matrix, 1.0)
    distance_matrix = 1 - coassoc_matrix
    np.fill_diagonal(distance_matrix, 0.0)
    linked = linkage(squareform(distance_matrix), method='average')
    consensus_labels = fcluster(linked, n_clusters, criterion='maxclust')
    return consensus_labels