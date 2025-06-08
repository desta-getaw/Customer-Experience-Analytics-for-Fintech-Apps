import pytest
from scripts.scrape_reviews import ReviewScraper

@pytest.fixture
def test_scraper():
    app_ids = {"Test Bank": "com.google.android.gms"}  # Google Play Services as stable test app
    languages = ['en']
    output_path = 'tests/test_output.csv'
    return ReviewScraper(app_ids, languages, output_path)

def test_scrape_returns_dataframe(test_scraper):
    df = test_scraper.scrape_all()
    assert not df.empty, "Scraped DataFrame is empty"
    assert 'review_text' in df.columns
    assert 'rating' in df.columns
    assert 'bank_name' in df.columns
