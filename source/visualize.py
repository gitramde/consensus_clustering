import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot(vect_dataset, consensus_labels, true_labels):
    # Reduce to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(vect_dataset)
    # Plot consensus cluster results
    plt.figure(figsize=(10, 5))

    # Plot predicted clusters
    plt.subplot(1, 2, 1)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=consensus_labels, cmap='tab10', s=10)
    plt.title("Consensus Clustering (Predicted)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # Plot true labels (if available)
    plt.subplot(1, 2, 2)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=true_labels, cmap='tab10', s=10)
    plt.title("True Newsgroup Labels")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    plt.tight_layout()
    plt.show()