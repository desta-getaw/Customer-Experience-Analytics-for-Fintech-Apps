# scripts/scrape_reviews.py

import os
import pandas as pd
from google_play_scraper import reviews, Sort

class ReviewScraper:
    def __init__(self, app_ids: dict, languages: list, output_path: str):
        self.app_ids = app_ids
        self.languages = languages
        self.output_path = output_path
        self.all_reviews = []

    def _scrape_app_reviews(self, app_id: str, bank_name: str, lang_code: str, count: int = 500):
        print(f"Fetching {lang_code.upper()} reviews for {bank_name}...")
        try:
            result, _ = reviews(
                app_id,
                lang=lang_code,
                country='et',
                sort=Sort.NEWEST,
                count=count
            )
            for r in result:
                self.all_reviews.append({
                    'review_text': r['content'],
                    'rating': r['score'],
                    'date': r['at'],
                    'bank_name': bank_name,
                    'source': 'Google Play',
                    'language': lang_code
                })
        except Exception as e:
            print(f"Error: {e}")

    def scrape_all(self):
        for bank, app_id in self.app_ids.items():
            for lang in self.languages:
                self._scrape_app_reviews(app_id, bank, lang)
        df = pd.DataFrame(self.all_reviews)
        return df

    def save_to_csv(self, df: pd.DataFrame):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False, encoding='utf-8-sig')
        print(f"Saved to {self.output_path}")
