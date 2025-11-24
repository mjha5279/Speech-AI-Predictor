import streamlit as st
import pandas as pd
import re
from sentence_transformers_lite import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pyspellchecker import SpellChecker
import nltk

nltk.download("punkt")

RUBRIC_PATH = "Rubric.xlsx"   # file name EXACT
SHEET_NAME = "Rubrics"        # sheet name EXACT
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"

# Load light model (no torch required)
model = SentenceTransformer(MODEL_NAME)

vader = SentimentIntensityAnalyzer()
spell = SpellChecker()

rub_df = pd.read_excel(RUBRIC_PATH, sheet_name=SHEET_NAME, engine="openpyxl")
rubric = rub_df.fillna("").to_dict(orient="records")

_word_re = re.compile(r"\w+['-]?\w*|\w+")

def tokenize(text):
    return _word_re.findall(text.lower())

def sentences(text):
    return [s.strip() for s in re.split(r"[.!?]\s*", text) if s.strip()]

def keyword_fraction(keywords, text_tokens):
    if not keywords:
        return 0.0
    keys = [k.strip().lower() for k in re.split(r"[;,|]", keywords) if k.strip()]
    txt = " ".join(text_tokens)
    found = sum(1 for k in keys if re.search(rf"\b{re.escape(k)}\b", txt))
    return found / len(keys) if keys else 0.0

def wordcount_score(words, min_w, max_w):
    if min_w and max_w:
        if min_w <= words <= max_w:
            return 1.0
        if words < min_w:
            return max(0.0, 1 - (min_w - words)/max(1, min_w))
        return max(0.0, 1 - (words - max_w)/max(1, max_w))
    if min_w:
        return 1.0 if words >= min_w else max(0.0, 1 - (min_w - words)/max(1, min_w))
    if max_w:
        return 1.0 if words <= max_w else max(0.0, 1 - (words - max_w)/max(1, max_w))
    return 1.0

def semantic_similarity(desc, transcript):
    if not desc or not transcript:
        return 0.0
    emb1 = model.encode([desc])
    emb2 = model.encode([transcript])
    sim = cosine_similarity([emb1[0]], [emb2[0]])[0][0]
    return max(0.0, min(1.0, (sim + 1)/2))

st.title("🎤 Speech Rubric Scorer")

transcript = st.text_area("Paste Transcript", height=250)
duration = st.number_input("Speech Duration (seconds)", min_value=1)

if st.button("Calculate Score"):
    if not transcript.strip():
        st.error("Please paste transcript.")
        st.stop()

    t = transcript.strip()
    tokens = tokenize(t)
    words = len(tokens)
    sent_count = len(sentences(t))
    wpm = words * 60 / duration

    # grammar
    miss = spell.unknown(tokens)
    grammar_errors = len(miss)
    errors_per_100 = (grammar_errors / words * 100) if words else 0
    grammar_score = 1 - min(errors_per_100 / 10, 1)

    # vocabulary
    ttr = len(set(tokens)) / words if words else 0

    # fillers
    fillers = ["um","uh","like","you know","so","actually","basically",
               "right","i mean","well","kinda","sort of","okay","hmm","ah"]
    filler_count = sum(len(re.findall(rf"\b{re.escape(f)}\b", t.lower())) for f in fillers)
    filler_rate_pct = (filler_count / words * 100) if words else 0

    # sentiment
    positive_prob = vader.polarity_scores(t)["pos"]

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

        kf = keyword_fraction(keywords, tokens)
        wc = wordcount_score(words, min_w, max_w)
        rule_score = 0.7*kf + 0.3*wc
        nlp_score = semantic_similarity(desc, t)
        final = (rule_score + nlp_score) / 2

        name_low = name.lower()

        if "grammar" in name_low: final = grammar_score
        if "vocab" in name_low: final = ttr
        if "filler" in name_low: final = 1 - (filler_rate_pct / 100)
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

    overall = (total_weighted / total_weights * 100) if total_weights else 0

    st.subheader("Final Score")
    st.metric("Score", f"{round(overall,2)}/100")

    st.subheader("Breakdown")
    st.dataframe(pd.DataFrame(per_rows))

    st.subheader("Details")
    st.json({
        "words": words,
        "sentences": sent_count,
        "wpm": wpm,
        "grammar_errors": grammar_errors,
        "errors_per_100_words": errors_per_100,
        "vocab_ttr": ttr,
        "filler_count": filler_count,
        "positive_sentiment": positive_prob
    })

