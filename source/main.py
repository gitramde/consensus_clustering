import time
import numpy as np
from data_extraction import data_load
from data_preparation import process
from clustering import kmeans_clustering
from visualize import plot
from coassociation import coassociation_matrix, hierarchial_clustering
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import TruncatedSVD

def core_module(consensus_approach):
    print("STEP 1: Extract Data")
    dl = data_load()
    newsgroup, categories = dl.get_data()
    data = newsgroup.data
    print("STEP 2: Process the Data")
    dp = process()
    vect_dataset1 = dp.vectorize_data(data)
    # Reduces high-dimensional TF-IDF vectors to a lower latent space
    # Makes K-Means behave better in sparse, noisy, high-dimensional spaces
    svd = TruncatedSVD(n_components=200, random_state=42)
    vect_dataset = svd.fit_transform(vect_dataset1)
    n_samples = vect_dataset.shape[0]
    n_clusters = len(categories)
    print("STEP 3: Partition Generation (Phase 1) using K-Means")
    start_time = time.monotonic()
    bc = kmeans_clustering(vect_dataset, n_clusters)
    end_time = time.monotonic()
    duration = end_time - start_time
    print("Elapsed Time in Seconds: ", duration)
    if consensus_approach == 'Coassociation':
        print("***** COASSOCIATION MATRIX METHOD ********")
        print("STEP 4: Building coassociation_matrix")
        start_time = time.monotonic()
        coassoc_matrix = coassociation_matrix(n_samples, bc)
        end_time = time.monotonic()
        duration = end_time - start_time
        print("Elapsed Time in Seconds: ", duration)
        print("STEP 5: Hierarchical Clutering on Coassociation Matrix:")
        start_time = time.monotonic()
        consensus_labels = hierarchial_clustering(coassoc_matrix, n_clusters)
        end_time = time.monotonic()
        duration = end_time - start_time
        print("Elapsed Time in Seconds: ", duration)
    elif consensus_approach == 'Median':
        print("***** MEDIAN PARTITIONING METHOD ********")
        print("STEP 4: Compute pairwise ARI similarity matrix")
        similarity_matrix = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                ari = adjusted_rand_score(bc[i], bc[j])
                similarity_matrix[i, j] = ari
                similarity_matrix[j, i] = ari
        print("STEP 5: Find the most central partition")
        total_similarity = similarity_matrix.sum(axis=1)
        median_index = np.argmax(total_similarity)
        consensus_labels = bc[median_index]
        end_time = time.monotonic()
        duration = end_time - start_time
        print("Elapsed Time in Seconds: ", duration)
    print("STEP 6: Evaluate against ground truth")
    start_time = time.monotonic()
    true_labels = newsgroup.target
    ari_score = adjusted_rand_score(true_labels, consensus_labels)
    print(ari_score)
    end_time = time.monotonic()
    duration = end_time - start_time
    print("Elapsed Time in Seconds: ", duration)
    plot(vect_dataset, consensus_labels, true_labels)

if __name__ == '__main__':
    core_module('Median')









