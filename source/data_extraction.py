import logging
from sklearn.datasets import fetch_20newsgroups

logger = logging.getLogger(__name__)

class data_load:
    _instance = None

    # Singleton Pattern to just create one instance
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.newsgroup = fetch_20newsgroups(subset='all', remove=('footers','quotes'))
        self.text = self.newsgroup.data
        self.labels = self.newsgroup.target
        self.target_names = self.newsgroup.target_names

    def get_labels(self):
        return self.labels

    def get_data(self):
        return self.text