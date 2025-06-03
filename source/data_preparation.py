from sklearn.feature_extraction.text import TfidfVectorizer
from config import parameters

class process:
    def vectorize_data(self, text):
        vectorizer = TfidfVectorizer( max_df=parameters.tdidf_max_df
                             ,min_df=parameters.tdidf_min_df
                             ,stop_words='english')

        vect_dataset = vectorizer.fit_transform(text)
        return vect_dataset
