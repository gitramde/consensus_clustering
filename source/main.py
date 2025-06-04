from data_extraction import data_load
from data_preparation import process
from sklearn.metrics import adjusted_rand_score
from coassociation import agglomerative

def model_evaluation(consensus_labels, true_labels):
    ari = adjusted_rand_score(true_labels, consensus_labels)
    print(f"Adjusted Rand Index (Consensus vs True Labels): {ari:.3f}")

if __name__ == '__main__':
    dl = data_load()
    dp = process()
    text = dl.get_data()[:50]
    vect_dataset = dp.vectorize_data(text)
    print("***** COASSOCIATION MATRIX METHOD ********")
    consensus_labels = agglomerative(vect_dataset)
    true_labels = dl.get_labels()[:50]
    model_evaluation(consensus_labels, true_labels)








