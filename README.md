🎤 Speech AI Predictor — Automated Rubric-Based Speech Scoring

This project is a lightweight NLP-powered speech evaluation system built with Streamlit.
It analyzes a written transcript of a speech and scores it according to a customizable rubric stored in Rubric.xlsx.

No heavy ML frameworks (no torch, no transformers) — fully deployable on Render or any lightweight hosting.

🚀 Features
✅ Rubric-Based Scoring

Each row in Rubric.xlsx defines:

Criterion name

Description

Keywords

Min/max word limits

Weight
You can modify this file to change the scoring logic without touching the code.

✅ Automated NLP Analysis

The app calculates:

Keyword coverage

Grammar errors (spell-check based)

Vocabulary richness (TTR)

Filler words

Sentiment score

Speech rate (WPM)

Semantic similarity using TF-IDF

✅ Detailed Output

Final score (0–100)

Individual criterion breakdown

Diagnostic information
(word count, WPM, grammar errors, sentiment, etc.)

✅ Fast & Deployable

Works on Render because it uses only lightweight libraries:

scikit-learn

numpy

nltk

pyspellchecker

pandas

📁 Project Structure
├── app.py              # Streamlit app
├── Rubric.xlsx         # Main scoring rubric (must be in root folder)
├── requirements.txt    # All dependencies for deployment
└── README.md           # Project documentation

🧠 How It Works

User pastes transcript

The app tokenizes text → extracts words, sentences, fillers, misspellings

For each rubric criterion:

Keyword match → score

Word count check → score

Semantic similarity → TF-IDF cosine score

Rule overrides for:
grammar, filler words, vocabulary, sentiment, speech rate

Weighted score computed → normalized to 0–100

UI displays:

Score

Table of all criteria

Debug insights

🛠️ Installation Instructions
1️⃣ Clone the repository
git clone https://github.com/mjha5279/Speech-AI-Predictor.git
cd Speech-AI-Predictor

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit
streamlit run app.py

☁️ Deployment (Render)

Push your repo to GitHub

Go to Render → New Web Service

Select this repo

Set:

Build Command: pip install -r requirements.txt

Start Command: streamlit run app.py --server.port $PORT --server.address 0.0.0.0

Deploy

📊 Rubric Format (Rubric.xlsx)

Your Excel sheet must contain a sheet named:

Rubrics


And columns:

criterion_name	description	keywords	min_words	max_words	weight

You can add as many rows as you want.
Every criterion is automatically processed.

📷 Screenshots (Optional)

You can add screenshots here later like:

![App Screenshot](images/app.png)

🙌 Credits

Developed using:

Streamlit

Scikit-learn

NLTK

VaderSentiment

PySpellChecker

📬 Contact

