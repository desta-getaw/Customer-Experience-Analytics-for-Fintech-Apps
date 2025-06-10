import pandas as pd
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import argparse

# Load distilBERT sentiment pipeline
bert_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Load VADER
vader_analyzer = SentimentIntensityAnalyzer()

def classify_distilbert(text):
    result = bert_classifier(text[:512])[0]
    label = result['label'].lower()
    score = result['score']
    if label == 'negative':
        return 'negative', score
    elif label == 'positive':
        return 'positive', score
    else:
        return 'neutral', score

def classify_vader(text):
    scores = vader_analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 'positive', compound
    elif compound <= -0.05:
        return 'negative', compound
    else:
        return 'neutral', compound

def classify_textblob(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0.05:
        return 'positive', score
    elif score < -0.05:
        return 'negative', score
    else:
        return 'neutral', score

def apply_sentiment_analysis(df, text_col='review'):
    sentiments = []
    for text in df[text_col]:
        try:
            label, score = classify_distilbert(text)
        except:
            label, score = 'neutral', 0.0
        sentiments.append((label, score))
    df['sentiment_label'], df['sentiment_score'] = zip(*sentiments)
    return df

def compare_methods(df, text_col='review'):
    bert_sent, vader_sent, textblob_sent = [], [], []
    for text in df[text_col]:
        b_lbl, b_score = classify_distilbert(text)
        v_lbl, v_score = classify_vader(text)
        t_lbl, t_score = classify_textblob(text)

        bert_sent.append((b_lbl, b_score))
        vader_sent.append((v_lbl, v_score))
        textblob_sent.append((t_lbl, t_score))

    df['bert_label'], df['bert_score'] = zip(*bert_sent)
    df['vader_label'], df['vader_score'] = zip(*vader_sent)
    df['textblob_label'], df['textblob_score'] = zip(*textblob_sent)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/cleaned_reviews.csv", help="Path to input CSV")
    parser.add_argument("--output", default="data/sentiment_reviews.csv", help="Path to save results")
    parser.add_argument("--compare", action="store_true", help="Run VADER and TextBlob comparison")

    args = parser.parse_args()

    print("Reading data...")
    df = pd.read_csv(args.input)

    if args.compare:
        print("Running sentiment comparison (BERT, VADER, TextBlob)...")
        df = compare_methods(df)
    else:
        print("Running sentiment analysis with distilBERT...")
        df = apply_sentiment_analysis(df)

    print(f"Saving results to {args.output}...")
    df.to_csv(args.output, index=False)
    print("Done.")
    print("Sentiment analysis completed.")
    print("Results saved successfully.")