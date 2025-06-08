# scripts/preprocess_reviews.py

import pandas as pd

class ReviewPreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def drop_missing(self):
        initial = len(self.df)
        self.df.dropna(subset=['review_text', 'rating', 'date'], inplace=True)
        print(f"Dropped {initial - len(self.df)} missing rows")

    def drop_duplicates(self):
        initial = len(self.df)
        self.df.drop_duplicates(subset=['review_text', 'rating', 'date', 'bank_name', 'language'], inplace=True)
        print(f"Dropped {initial - len(self.df)} duplicates")

    def normalize_dates(self):
        initial = len(self.df)
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        self.df.dropna(subset=['date'], inplace=True)
        print(f"Dropped {initial - len(self.df)} bad dates")
        self.df['date'] = self.df['date'].dt.strftime('%Y-%m-%d')

    def prepare_columns(self):
        self.df.rename(columns={
            'review_text': 'review',
            'bank_name': 'bank'
        }, inplace=True)
        self.df = self.df[['review', 'rating', 'date', 'bank', 'source', 'language']]

    def get_cleaned_data(self):
        return self.df
