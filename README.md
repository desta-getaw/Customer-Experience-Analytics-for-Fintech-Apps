# 📊 Fintech Mobile App Review Analysis

This project is a real-world data engineering and analysis challenge designed for 10 Academy's Artificial Intelligence Mastery Week 2.  
We analyze customer reviews from the Google Play Store for Ethiopian banks using web scraping, NLP, and visualization techniques, all backed by CI/CD automation with GitHub Actions.

---

## 🚀 Project Objectives

- Scrape reviews from the Google Play Store for **Commercial Bank of Ethiopia (CBE)**, **Bank of Abyssinia (BOA)**, and **Dashen Bank**.
- Clean and preprocess the review data.
- Perform sentiment and thematic analysis using NLP techniques.
- Store cleaned data in an Oracle database.
- Visualize insights and provide actionable recommendations.
- Automate testing and linting via GitHub Actions.

---

## 🗂 Project Structure

```bash
fintech-review-analysis/
├── .github/workflows/ci.yml         # GitHub Actions workflow for CI
├── data/                            # Folder for raw scraped data
├── output/                          # Folder for cleaned/processed data
├── scripts/                         # Python scripts for scraping, preprocessing, NLP
├── tests/                           # Unit test files
├── requirements.txt                 # All required packages
├── main.py                          # Entry point (can orchestrate the pipeline)
└── README.md                        # Project documentation
```

---

## ⚙️ Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/fintech-review-analysis.git
   cd fintech-review-analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main pipeline:**
   ```bash
   python main.py
   ```

---

## 🧪 Testing

We use `pytest` for unit testing. To run all tests:

```bash
pytest tests/
```

---

## ✅ CI/CD with GitHub Actions

Our CI pipeline performs the following on each push to `main`:
- Installs dependencies
- Runs `flake8` for linting
- Executes unit tests
- Uploads cleaned CSVs as artifacts

See configuration in `.github/workflows/ci.yml`.

---

## 📦 Features to Implement

- `scripts/scrape_reviews.py` — Scrape reviews using `google-play-scraper`
- `scripts/preprocess.py` — Clean text, handle missing/duplicate data
- `scripts/sentiment_analysis.py` — Analyze sentiment with TextBlob or Hugging Face
- `scripts/db_insert.py` — Insert reviews into Oracle or PostgreSQL
- `main.py` — Orchestrate the complete pipeline

---

## 📊 Visualization (Task 4)

Use `matplotlib` or `seaborn` to generate:
- Sentiment distribution plots
- Word clouds for each bank
- Top themes/issues bar charts

---

## 🔍 Useful References

- [google-play-scraper](https://pypi.org/project/google-play-scraper/)
- [TextBlob Sentiment](https://textblob.readthedocs.io/en/dev/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Oracle DB Python Driver](https://python-oracledb.readthedocs.io/en/latest/)

---

## 🧑‍💻 Contributors

- This project is part of **10 Academy Week 2 Challenge**
- Mentors: Mahlet, Kerod, Rediet, Rehmet

---

## 📅 Deadlines

- **June 08, 2024** – Interim submission (2 pages, GitHub link with Task 1)
- **June 10, 2024** – Final submission (7 pages, all tasks + plots)

---

## 💬 Contact

Questions? Reach out via Slack: `#all-week-2` or contact your facilitators.
