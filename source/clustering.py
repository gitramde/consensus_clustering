from sklearn.cluster import KMeans
from config import parameters
def kmeans_clustering(vect_dataset):
    base_clustering =[]
    for i in range(parameters.n_base_clusterings):
        km = KMeans(n_clusters=parameters.n_clusters,
                    init='random',
                    n_init=parameters.n_init,
                    random_state=i)
        labels = km.fit_predict(vect_dataset)
        base_clustering.append(labels)
    return base_clustering
