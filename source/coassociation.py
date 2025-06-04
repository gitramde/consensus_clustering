import numpy as np
import time
from config import parameters
from sklearn.cluster import AgglomerativeClustering
from clustering import kmeans_clustering

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

def agglomerative(vect_dataset):
    """
    Consesus clustering using agglomerative on coassociation matrix
    Similarity is converted to distance by subtracting from 1
    """
    n_samples = vect_dataset.shape[0]
    print("STEP 1: Calling base_clustering")
    start_time = time.monotonic()
    bc = kmeans_clustering(vect_dataset)
    end_time = time.monotonic()
    duration = end_time - start_time
    print("Elapsed Time in Seconds: ", duration)
    print("STEP 2: Calling coassociation_matrix")
    start_time = time.monotonic()
    coassoc=coassociation_matrix(n_samples, bc)
    end_time = time.monotonic()
    duration = end_time - start_time
    print("Elapsed Time in Seconds: ", duration)
    print("STEP 3: Calling AgglomerativeClustering")
    start_time = time.monotonic()
    consensus = AgglomerativeClustering(n_clusters= parameters.n_clusters,
                                        metric='precomputed',
                                        linkage='average')
    end_time = time.monotonic()
    duration = end_time - start_time
    print("Elapsed Time in Seconds: ", duration)
    print("STEP 4: Calling fit_predict")
    start_time = time.monotonic()
    distance_matrix = 1 - coassoc
    consensus_labels = consensus.fit_predict(distance_matrix)
    end_time = time.monotonic()
    duration = end_time - start_time
    print("Elapsed Time in Seconds: ", duration)
    return consensus_labels