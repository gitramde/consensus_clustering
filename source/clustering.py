import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import parameters
from data_extraction import data_load
from sklearn.cluster import KMeans

# Initialize consensus matrix

consensus_matrix = np.zeros((data_load.n_samples, data_load.n_samples))

# Track how often each sample is used
sample_counts = np.zeros((data_load.n_samples, data_load.n_samples))

# Run clustering multiple times

for i in range(parameters.n_iterations):
    sample_indices = np.random.choice(data_load.n_samples
                                      ,size=int(data_load.n_samples * parameters.sample_fraction)
                                      ,replace=False)
    X_sub = data_load.iris_data[sample_indices]


    # KMeans clustering
    kmeans = KMeans(n_clusters=parameters.n_clusters,
                n_init=10,
                random_state=i)

    labels = kmeans.fit_predict(X_sub)

    # Map cluster label to full matrix

    for i1, idx1 in enumerate(sample_indices):
        for i2, idx2 in enumerate(sample_indices):
            sample_counts[idx1, idx2] == 1
            if labels[i1] == labels[i2]:
                consensus_matrix[idx1, idx2] += 1

    # Normalize Consensus matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        consensus_matrix = np.true_divide(consensus_matrix, sample_counts)
        consensus_matrix[sample_counts == 0 ] = 0

    # Visualize consensus matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(consensus_matrix, cmap='viridis')
        plt.title("Consensus Matrix heatmap")
        plt.xlabel("Sample Index")
        plt.ylabel("Sample Index")
        plt.show()

    # Final clustering using hierarchical clustering on consensus matrix
    # convert consensus similarity to distance

    distance_matrix = 1 - consensus_matrix
    linkage_matrix = linkage(pairwise_distances(consensus_matrix), method='average')
    final_labels = fcluster(linkage_matrix, parameters.n_clusters, criterion = 'maxclust')


    #Plot final cluster assignments
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2 )
    x_pca = pca.fit_transform(data_load.iris_data)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_pca[:, 0]
                , x_pca[:, 1]
                , c=final_labels
                , cmap = 'Set1'
                , s=50)
    plt.title("Final Clusters from Consensus Clustering")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()




