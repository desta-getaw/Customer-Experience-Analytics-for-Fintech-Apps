import pandas as pd
import pytest
from scripts.preprocess_reviews import ReviewPreprocessor

@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {'review_text': 'Great app!', 'rating': 5, 'date': '2024-01-01', 'bank_name': 'Test Bank', 'source': 'Google Play', 'language': 'en'},
        {'review_text': 'Great app!', 'rating': 5, 'date': '2024-01-01', 'bank_name': 'Test Bank', 'source': 'Google Play', 'language': 'en'},
        {'review_text': None, 'rating': 4, 'date': '2024-02-02', 'bank_name': 'Test Bank', 'source': 'Google Play', 'language': 'en'},
        {'review_text': 'Bad app', 'rating': 1, 'date': 'invalid-date', 'bank_name': 'Test Bank', 'source': 'Google Play', 'language': 'en'}
    ])

def test_preprocessor_cleaning(sample_df):
    pre = ReviewPreprocessor(sample_df)
    pre.drop_missing()
    pre.drop_duplicates()
    pre.normalize_dates()
    pre.prepare_columns()
    clean_df = pre.get_cleaned_data()

    assert not clean_df.empty, "Cleaned DataFrame is empty"
    assert clean_df.shape[0] == 1, "Unexpected number of rows after cleaning"
    assert all(col in clean_df.columns for col in ['review', 'rating', 'date', 'bank', 'source', 'language'])
