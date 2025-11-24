import streamlit as st
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from spellchecker import SpellChecker
import nltk

# Download tokenizer
nltk.download('punkt')

# -------- CONFIG ----------
RUBRIC_PATH = "rubric.xlsx"

vader = SentimentIntensityAnalyzer()
spell = SpellChecker()

# Load rubric
rub_df = pd.read_excel(RUBRIC_PATH, sheet_name="Rubrics", engine="openpyxl")
rubric = rub_df.fillna("").to_dict(orient="records")

# ------- HELPERS ----------
_word_re = re.compile(r"\w+['-]?\w*|\w+")

def tokenize(text):
    return _word_re.findall(text.lower())

def sentences(text):
    return [s.strip() for s in re.split(r'[.!?]\s*', text) if s.strip()]

def keyword_fraction(keywords, text_tokens):
    if not keywords:
        return 0.0
    keys = [k.strip().lower() for k in re.split(r'[;,|]', keywords) if k.strip()]
    if not keys:
        return 0.0
    txt = " ".join(text_tokens)
    found = sum(1 for k in keys if re.search(r'\b' + re.escape(k) + r'\b', txt))
    return found / len(keys)

def wordcount_score(words, min_w, max_w):
    if min_w and max_w:
        if min_w <= words <= max_w: return 1.0
        if words < min_w: return max(0.0, 1 - (min_w - words)/max(1,min_w))
        return max(0.0, 1 - (words - max_w)/max(1,max_w))
    if min_w:
        return 1.0 if words >= min_w else max(0.0, 1 - (min_w - words)/max(1,min_w))
    if max_w:
        return 1.0 if words <= max_w else max(0.0, 1 - (words - max_w)/max(1,max_w))
    return 1.0

# -------- STREAMLIT UI ----------
st.title("🎤 Speech Rubric Scorer")
st.write("Paste transcript below and get automated scoring using NLP + Rubric.")

transcript = st.text_area("Paste Transcript", height=250)
duration = st.number_input("Speech Duration (seconds)", min_value=1)

if st.button("Calculate Score"):
    if not transcript.strip():
        st.error("Please paste transcript.")
    else:
        t = transcript.strip()
        tokens = tokenize(t)
        words = len(tokens)
        sents = sentences(t)
        sent_count = len(sents)
        wpm = words * 60 / duration

        # Grammar
        miss = spell.unknown(tokens)
        grammar_errors = len(miss)
        errors_per_100 = (grammar_errors / words * 100) if words > 0 else 0
        grammar_score_val = 1 - min(errors_per_100 / 10.0, 1.0)

        # Vocabulary
        distinct = len(set(tokens))
        ttr = distinct / words if words > 0 else 0.0

        # Filler words
        fillers = ["um","uh","like","you know","so","actually","basically",
                   "right","i mean","well","kinda","sort of","okay","hmm","ah"]
        filler_count = sum(len(re.findall(r'\b' + re.escape(f) + r'\b', t.lower())) for f in fillers)
        filler_rate_pct = (filler_count / words * 100) if words > 0 else 0

        # Sentiment
        sent_scores = vader.polarity_scores(t)
        positive_prob = sent_scores["pos"]

        # Per-criterion scoring
        per_rows = []
        total_weighted = 0
        total_weights = 0

        for r in rubric:
            name = r.get("criterion_name") or "Unnamed"
            desc = str(r.get("description", ""))
            keywords = str(r.get("keywords", ""))
            weight = float(r.get("weight", 0))
            min_w = int(r.get("min_words")) if r.get("min_words") else None
            max_w = int(r.get("max_words")) if r.get("max_words") else None

            k_frac = keyword_fraction(keywords, tokens)
            wc = wordcount_score(words, min_w, max_w)

            # Combine keyword + wordcount (no semantic model)
            final = 0.7*k_frac + 0.3*wc

            name_low = name.lower()

            # Overrides
            if "grammar" in name_low: final = grammar_score_val
            if "vocab" in name_low or "vocabulary" in name_low: final = ttr
            if "filler" in name_low: final = max(0.0, 1 - (filler_rate_pct / 100))

            if "speech rate" in name_low or "wpm" in name_low:
                if wpm > 161: final = 0.2
                elif 141 <= wpm <= 160: final = 0.6
                elif 111 <= wpm <= 140: final = 1.0
                elif 81 <= wpm <= 110: final = 0.6
                else: final = 0.2

            if "sentiment" in name_low: final = positive_prob

            score_numeric = final * weight

            per_rows.append({
                "Criterion": name,
                "Score (0-1)": round(final, 3),
                "Weight": weight,
                "Weighted Score": round(score_numeric, 3)
            })

            total_weighted += score_numeric
            total_weights += weight

        overall_norm = total_weighted / total_weights if total_weights > 0 else 0
        overall_100 = round(overall_norm * 100, 2)

        st.subheader("Overall Score")
        st.metric("Total", f"{overall_100}/100")

        st.subheader("Criterion Breakdown")
        st.dataframe(pd.DataFrame(per_rows))

        st.subheader("Details")
        st.json({
            "words": words,
            "sentences": sent_count,
            "wpm": wpm,
            "grammar_errors": grammar_errors,
            "errors_per_100": errors_per_100,
            "ttr_vocabulary_score": ttr,
            "filler_count": filler_count,
            "positive_sentiment": positive_prob
        })

