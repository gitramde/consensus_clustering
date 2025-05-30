from sklearn.datasets import load_iris

class data_load():
    iris = load_iris()
    iris_data = iris.data
    n_samples = iris_data.shape[0]