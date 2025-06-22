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
        self.categories = ['comp.graphics', 'rec.sport.baseball', 'sci.space', 'talk.politics.mideast']
        self.newsgroup = fetch_20newsgroups(subset='train',
                                            categories = self.categories,
                                            remove=('footers','quotes'))
        self.labels = self.newsgroup.target
        self.target_names = self.newsgroup.target_names

    def get_labels(self):
        return self.labels

    def get_data(self):
        return self.newsgroup, self.categories

if __name__ == "__main__":
    dl = data_load()
    data = dl.get_data()
    print(type(data))
    print(len(data))