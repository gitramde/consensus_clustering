import numpy as np
import time
from data_extraction import data_load
from data_preparation import process
from clustering import kmeans_clustering
from config import parameters
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
# Load Dataset
def vectorize_data(text):
    """
    TF-IDF vectorization - Evaluate the importance
    of a word within a document relative to a collection
    of documents, also known as a corpus
    """
    vect_dataset = dp.vectorize_data(text)
    return vect_dataset

def base_clustering(vect_dataset, type):
    """
    This function generates multiple KMeans clusterings known as
    base_clustering
    """
    if type == 'KMeans':
        bc = kmeans_clustering(vect_dataset)
    return bc

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

def agglomerative(text, type):
    """
    Consesus clustering using agglomerative on coassociation matrix
    Similarity is converted to distance by subtracting from 1
    """
    start_time = time.monotonic()
    print("STEP 1: Calling vectorize_data")
    vect_dataset = vectorize_data(text)
    n_samples = vect_dataset.shape[0]
    end_time = time.monotonic()
    duration = end_time - start_time
    print("Elapsed Time in Seconds: ", duration)
    print("STEP 2: Calling base_clustering")
    start_time = time.monotonic()
    bc = base_clustering(vect_dataset, type)
    end_time = time.monotonic()
    duration = end_time - start_time
    print("Elapsed Time in Seconds: ", duration)
    print("STEP 3: Calling coassociation_matrix")
    start_time = time.monotonic()
    coassoc=coassociation_matrix(n_samples, bc)
    #sample_idx = np.random.choice(n_samples, 2000, replace=False)
    #coassoc_sample = coassoc[np.ix_(sample_idx, sample_idx)]
    #distance_matrix = 1 - coassoc_sample
    end_time = time.monotonic()
    duration = end_time - start_time
    print("Elapsed Time in Seconds: ", duration)
    print("STEP 4: Calling AgglomerativeClustering")
    start_time = time.monotonic()
    consensus = AgglomerativeClustering(n_clusters= parameters.n_clusters,
                                        metric='precomputed',
                                        linkage='average')
    end_time = time.monotonic()
    duration = end_time - start_time
    print("Elapsed Time in Seconds: ", duration)
    print("STEP 5: Calling fit_predict")
    start_time = time.monotonic()
    distance_matrix = 1 - coassoc
    consensus_labels = consensus.fit_predict(distance_matrix)
    end_time = time.monotonic()
    duration = end_time - start_time
    print("Elapsed Time in Seconds: ", duration)
    return consensus_labels

def model_evaluation(consensus_labels, true_labels):
    ari = adjusted_rand_score(true_labels, consensus_labels)
    print(f"Adjusted Rand Index (Consensus vs True Labels): {ari:.3f}")

if __name__ == '__main__':
    dl = data_load()
    dp = process()
    text = dl.get_data()
    print("Calling agglomerative in main")
    consensus_labels = agglomerative(text, 'KMeans')
    true_labels = dl.get_labels()
    model_evaluation(consensus_labels, true_labels)








