# scripts/keywords.py
#pip install pandas scikit-learn spacy
#python -m spacy download en_core_web_sm

##
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from collections import defaultdict

class KeywordExtractor:
    def __init__(self, method='tfidf', top_n=20):
        self.method = method
        self.top_n = top_n
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    def extract_tfidf(self, texts):
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            max_features=500
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        tfidf_scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.sum(axis=0).A1)
        sorted_keywords = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        return sorted_keywords[:self.top_n]

    def extract_spacy_noun_phrases(self, texts):
        phrase_freq = defaultdict(int)
        docs = self.nlp.pipe(texts)
        for doc in docs:
            for chunk in doc.noun_chunks:
                phrase_freq[chunk.text.lower()] += 1
        sorted_phrases = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)
        return sorted_phrases[:self.top_n]

    def extract_keywords(self, texts):
        if self.method == 'tfidf':
            return self.extract_tfidf(texts)
        elif self.method == 'spacy':
            return self.extract_spacy_noun_phrases(texts)
        else:
            raise ValueError("Method must be 'tfidf' or 'spacy'.")

    def manual_cluster(self, keywords):
        clusters = defaultdict(list)
        for kw, _ in keywords:
            kw_lower = kw.lower()
            if any(word in kw_lower for word in ["login", "auth", "password"]):
                clusters["Login Issues"].append(kw)
            elif any(word in kw_lower for word in ["slow", "delay", "freeze", "lag"]):
                clusters["Performance"].append(kw)
            elif any(word in kw_lower for word in ["ui", "interface", "layout", "design"]):
                clusters["UI Feedback"].append(kw)
            elif any(word in kw_lower for word in ["transfer", "send", "transaction", "payment"]):
                clusters["Transactions"].append(kw)
            elif any(word in kw_lower for word in ["crash", "bug", "error"]):
                clusters["App Errors"].append(kw)
            else:
                clusters["Other"].append(kw)
        return dict(clusters)
