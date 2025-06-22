from sklearn.cluster import KMeans
from config import parameters
import random
def kmeans_clustering(vect_dataset, n_clusters):
    base_clustering =[]
    #k_range = [4, 5] # Improve the partition diversity
    for i in range(parameters.n_base_clusterings):
        #k = random.choice(k_range)
        km = KMeans(n_clusters=n_clusters,
                    init='k-means++',
                    random_state=i)
        labels = km.fit_predict(vect_dataset)
        base_clustering.append(labels)
    return base_clustering
