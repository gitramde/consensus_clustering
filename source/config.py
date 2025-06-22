class parameters():

    n_iterations = 100
    sample_fraction = 0.8
    n_runs = 30
    # TF-IDF parameters
    tdidf_max_df = 0.7 # Remove overly common/stopword-like terms
    tdidf_min_df = 5 # Remove noisy rare words
    max_features = 10000 # Focus on top relevant words

    #K-Means Clustering
    n_base_clusterings = 30 # Defining Partitions for the first phase of Consensus Clustering
    n_init = 1


