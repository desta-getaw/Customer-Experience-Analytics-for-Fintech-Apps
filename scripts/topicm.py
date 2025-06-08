# scripts/topic_modeling.py
import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from typing import List, Tuple

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

class TopicModeler:
    def __init__(self, max_features: int = 1000, n_topics: int = 5):
        self.max_features = max_features
        self.n_topics = n_topics
        self.vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=max_features)
        self.lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and len(token) > 2]
        return " ".join(tokens)

    def preprocess_reviews(self, df: pd.DataFrame, text_column: str = "review") -> pd.DataFrame:
        df["cleaned_review"] = df[text_column].fillna("").apply(self.clean_text)
        return df

    def vectorize_text(self, corpus: List[str]) -> any:
        doc_term_matrix = self.vectorizer.fit_transform(corpus)
        return doc_term_matrix

    def apply_lda(self, doc_term_matrix: any) -> LatentDirichletAllocation:
        self.lda_model.fit(doc_term_matrix)
        return self.lda_model

    def get_topic_keywords(self, n_top_words: int = 10) -> List[List[str]]:
        words = self.vectorizer.get_feature_names_out()
        topics = []
        for topic in self.lda_model.components_:
            top_words = [words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            topics.append(top_words)
        return topics

    def assign_dominant_topic(self, doc_term_matrix: any) -> List[int]:
        topic_distribution = self.lda_model.transform(doc_term_matrix)
        dominant_topics = topic_distribution.argmax(axis=1)
        return dominant_topics
