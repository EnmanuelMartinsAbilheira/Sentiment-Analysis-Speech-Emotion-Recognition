from flask import Flask, render_template, request
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from spellchecker import SpellChecker
from gramformer import Gramformer

app = Flask(__name__)

# Load models and vectorizer
with open("Project 3\\input_view\\models\\best_model_Type.pkl", "rb") as f:
    type_model = pickle.load(f)

with open("Project 3\\input_view\\models\\best_model_Factuality.pkl", "rb") as f:
    fact_model = pickle.load(f)

with open("Project 3\\input_view\\models\\best_model_Sentiment.pkl", "rb") as f:
    sentiment_model = pickle.load(f)

with open("Project 3\\input_view\\models\\tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load sentence encoder
sentence_encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Mappings from your notebook
type_mapping = {0: "affirmative", 1: "negation"}
fact_subj_mapping = {0: "fact", 1: "opinion"}
sentiment_mapping = {0: "sadness", 1: "anger", 2: "neutral", 3: "happiness", 4: "euphoria"}

def correct_spelling(sentence):
    spell_checker = SpellChecker()
    words = sentence.split() 
    corrected_words = [spell_checker.correction(word) or word for word in words] 
    result = " ".join(corrected_words)
    return result

def correct_sentence(sentence):
    gramformer = Gramformer(models=1, use_gpu=False)
    spelled_corrected = correct_spelling(sentence)
    
    corrected_sentences = gramformer.correct(spelled_corrected, max_candidates=1)
    result = next(iter(corrected_sentences), spelled_corrected)
    
    return result

def preprocess_text(text, model_type=None):
    """Preprocess text using either sentence embeddings or TF-IDF"""
    # Use TF-IDF for factuality model
    if model_type == 'factuality':
        return vectorizer.transform([text])
    
    # Try sentence embeddings for other models
    try:
        return sentence_encoder.encode([text]).reshape(1, -1)
    except Exception:
        # Fall back to TF-IDF if sentence embeddings fail
        return vectorizer.transform([text])

def model_predict(text):
    """Makes predictions using all three models"""
    # Type prediction
    try:
        type_features = preprocess_text(text, 'type')
        type_pred = type_model.predict(type_features)[0]
        type_label = type_mapping[type_pred]
    except Exception as e:
        print(f"Type prediction error: {e}")
        type_label = "Unknown"

    # Factuality prediction
    try:
        fact_features = preprocess_text(text, 'factuality')
        fact_pred = fact_model.predict(fact_features)[0]
        fact_label = fact_subj_mapping[fact_pred]
    except Exception as e:
        print(f"Factuality prediction error: {e}")
        fact_label = "Unknown"

    # Sentiment prediction
    try:
        sent_features = preprocess_text(text, 'sentiment')
        sent_pred = sentiment_model.predict(sent_features)[0]
        sent_label = sentiment_mapping[sent_pred]
    except Exception as e:
        print(f"Sentiment prediction error: {e}")
        sent_label = "Unknown"

    return type_label, fact_label, sent_label

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("text")
        corrected = correct_sentence(text)
        type_label, fact_label, sentiment_label = model_predict(text)

        return render_template(
            "result.html",
            original_text=text,
            corrected_text=corrected,
            type_pred=type_label,
            factual_pred=fact_label,
            sentiment_pred=sentiment_label
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
