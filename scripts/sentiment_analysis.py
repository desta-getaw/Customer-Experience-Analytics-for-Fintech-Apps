# scripts/sentiment.py

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline


class ReviewSentiment:
    def __init__(self, method='vader'):
        self.method = method
        self.analyzer = None

        if method == 'vader':
            self.analyzer = SentimentIntensityAnalyzer()
        elif method == 'distilbert':
            self.analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        else:
            raise ValueError("Unsupported sentiment method")

    def analyze_sentiment(self, texts):
        if self.method == 'vader':
            return [self._analyze_vader(text) for text in texts]
        elif self.method == 'distilbert':
            return [self._analyze_distilbert(text) for text in texts]

    def _analyze_vader(self, text):
        score = self.analyzer.polarity_scores(text)['compound']
        if score >= 0.05:
            return 1
        elif score <= -0.05:
            return -1
        else:
            return 0

    def _analyze_distilbert(self, text):
        result = self.analyzer(text)[0]
        return 1 if result['label'] == 'POSITIVE' else -1

    def apply(self, df, text_col='review'):
        df = df.copy()
        df['sentiment'] = self.analyze_sentiment(df[text_col])
        return df

    def aggregate(self, df, group_cols=['bank', 'rating']):
        return (
            df.groupby(group_cols)
              .agg(mean_sentiment=('sentiment', 'mean'),
                   review_count=('sentiment', 'count'))
              .reset_index()
        )
