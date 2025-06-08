# scripts/topic_modeling.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

class TopicModeler:
    def __init__(self, n_topics=5, model_type="lda"):
        self.n_topics = n_topics
        self.model_type = model_type
        self.model = None
        self.vectorizer = None

    def fit(self, documents):
        if self.model_type == "lda":
            self.vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words='english')
            doc_term_matrix = self.vectorizer.fit_transform(documents)
            self.model = LatentDirichletAllocation(n_components=self.n_topics, random_state=42)
        elif self.model_type == "nmf":
            self.vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, stop_words='english')
            doc_term_matrix = self.vectorizer.fit_transform(documents)
            self.model = NMF(n_components=self.n_topics, random_state=42)
        else:
            raise ValueError("Choose model_type as 'lda' or 'nmf'")
        self.model.fit(doc_term_matrix)
        return self.model, self.vectorizer

    def display_topics(self, n_words=10):
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for idx, topic in enumerate(self.model.components_):
            keywords = [feature_names[i] for i in topic.argsort()[-n_words:]]
            topics.append({"Topic": idx + 1, "Keywords": ", ".join(keywords)})
        return pd.DataFrame(topics)
