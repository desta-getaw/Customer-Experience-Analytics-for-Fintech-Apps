import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import spacy
import argparse

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# --- Define theme rules (manual) ---
THEME_RULES = {
    "Account Access Issues": ["login", "password", "authentication", "sign", "session"],
    "Transaction Performance": ["transfer", "slow", "fail", "timeout", "crash", "delay"],
    "User Interface & Experience": ["interface", "ui", "ux", "navigation", "layout", "design"],
    "Customer Support": ["support", "help", "assist", "response", "service"],
    "Feature Requests": ["feature", "add", "include", "missing", "option", "functionality"]
}

# --- Extract TF-IDF Keywords ---
def extract_tfidf_keywords(reviews, max_features=30, ngram_range=(1, 2)):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(reviews)
    return list(vectorizer.get_feature_names_out())

# --- Assign Themes Based on Rules ---
def assign_themes(text, theme_rules):
    text_lower = text.lower()
    matched = []
    for theme, keywords in theme_rules.items():
        if any(keyword in text_lower for keyword in keywords):
            matched.append(theme)
    return ', '.join(matched) if matched else "Other"

# --- Assign Themes to Reviews ---
def apply_theme_analysis(df, theme_rules):
    df['identified_themes'] = df['review'].apply(lambda x: assign_themes(str(x), theme_rules))
    return df

# --- Print Top Keywords Per Bank ---
def summarize_keywords_by_bank(df):
    banks = df['bank'].unique()
    for bank in banks:
        print(f"\n🔹 Top keywords for {bank}:")
        bank_reviews = df[df['bank'] == bank]['review'].dropna().tolist()
        keywords = extract_tfidf_keywords(bank_reviews, max_features=20)
        print(', '.join(keywords))

# --- Print Theme Distributions ---
def summarize_themes(df):
    print("\n📊 Theme distribution (all banks):")
    print(df['identified_themes'].value_counts())
    print("\n📊 Theme distribution by bank:")
    print(df.groupby('bank')['identified_themes'].value_counts())

# --- Main function ---
def run_thematic_analysis(input_path, output_path):
    print("📥 Loading data...")
    df = pd.read_csv(input_path)
    df.dropna(subset=['review'], inplace=True)

    print("🧠 Assigning themes...")
    df = apply_theme_analysis(df, THEME_RULES)

    print("📌 Extracting keywords per bank...")
    summarize_keywords_by_bank(df)

    print("📈 Generating theme summary...")
    summarize_themes(df)

    print(f"💾 Saving to: {output_path}")
    df.to_csv(output_path, index=False)
    print("✅ Done.")

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/sentiment_reviews.csv', help='Input CSV file path')
    parser.add_argument('--output', default='data/thematic_reviews.csv', help='Output CSV with themes')
    args = parser.parse_args()
    run_thematic_analysis(args.input, args.output)
