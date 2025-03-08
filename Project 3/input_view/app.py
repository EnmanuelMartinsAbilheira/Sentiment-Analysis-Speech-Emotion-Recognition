from flask import Flask, render_template, request
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from spellchecker import SpellChecker
from gramformer import Gramformer
import spacy
from transformers import pipeline

app = Flask(__name__)

"""Here we load the custom models"""
# Load models and vectorizer
with open("Project 3\\input_view\\models\\best_model_Type.pkl", "rb") as f:
    type_model = pickle.load(f)

with open("Project 3\\input_view\\models\\best_model_Factuality.pkl", "rb") as f:
    fact_model = pickle.load(f)

with open("Project 3\\input_view\\models\\best_model_Sentiment.pkl", "rb") as f:
    sentiment_model = pickle.load(f)

with open("Project 3\\input_view\\models\\tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

"""Here we load the pretrained models"""
# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize emotion classifier
print("Loading emotion classification model...")
emotion_classifier = pipeline("text-classification", 
model="ayoubkirouane/BERT-Emotions-Classifier", return_all_scores=True)

# Initialize specialized fact vs. opinion classifier with correct model name
print("Loading specialized fact-opinion classification model...")
fact_opinion_classifier = pipeline(
    "text-classification",
    model="lighteternal/fact-or-opinion-xlmr-el"
)

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

#Custom models
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

def predict_with_custom_model(text, prediction_type):
    """Makes predictions using the specified custom model"""
    if prediction_type == 'type':
        try:
            features = preprocess_text(text, 'type')
            pred = type_model.predict(features)[0]
            return type_mapping[pred]
        except Exception as e:
            print(f"Type prediction error: {e}")
            return "Unknown"
    
    elif prediction_type == 'factuality':
        try:
            features = preprocess_text(text, 'factuality')
            pred = fact_model.predict(features)[0]
            return fact_subj_mapping[pred]
        except Exception as e:
            print(f"Factuality prediction error: {e}")
            return "Unknown"
    
    elif prediction_type == 'sentiment':
        try:
            features = preprocess_text(text, 'sentiment')
            pred = sentiment_model.predict(features)[0]
            return sentiment_mapping[pred]
        except Exception as e:
            print(f"Sentiment prediction error: {e}")
            return "Unknown"
    
    return "Unknown"

#Pretrained models
def predict_with_pretrained_model(text, prediction_type):
    """Makes predictions using pretrained models"""
    if prediction_type == 'type':
        doc = nlp(text)
        has_negation = any(token.dep_ == 'neg' for token in doc)
        return "negation" if has_negation else "affirmative"
    
    elif prediction_type == 'factuality':
        result = fact_opinion_classifier(text)[0]
        label_map = {"LABEL_0": "opinion", "LABEL_1": "fact"}
        return label_map.get(result['label'], result['label'])
    
    elif prediction_type == 'sentiment':
        emotion_scores = emotion_classifier(text)[0]
        sorted_emotions = sorted(emotion_scores, key=lambda x: x['score'], reverse=True)
        return sorted_emotions[0]['label']
    
    return "Unknown"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("text")
        model_type = request.form.get("model_type", "custom")  # Get the selected model type
        
        # Correct spelling and grammar
        corrected = correct_sentence(text)
        
        # Initialize model choices based on the selected model_type
        type_model_choice = request.form.get("type_model", "custom")
        fact_model_choice = request.form.get("fact_model", "custom")
        sentiment_model_choice = request.form.get("sentiment_model", "custom")
        
        # Get type prediction
        if type_model_choice == "custom":
            type_pred = predict_with_custom_model(corrected, 'type')
            type_model_label = "Modelo Customizado"
        else:
            type_pred = predict_with_pretrained_model(corrected, 'type')
            type_model_label = "Modelo Pré-treinado"
        
        # Get factuality prediction
        if fact_model_choice == "custom":
            fact_pred = predict_with_custom_model(corrected, 'factuality')
            fact_model_label = "Modelo Customizado"
        else:
            fact_pred = predict_with_pretrained_model(corrected, 'factuality')
            fact_model_label = "Modelo Pré-treinado"
        
        # Get sentiment prediction
        if sentiment_model_choice == "custom":
            sentiment_pred = predict_with_custom_model(corrected, 'sentiment')
            sentiment_model_label = "Modelo Customizado"
        else:
            sentiment_pred = predict_with_pretrained_model(corrected, 'sentiment')
            sentiment_model_label = "Modelo Pré-treinado"

        return render_template(
            "result.html",
            original_text=text,
            corrected_text=corrected,
            type_pred=type_pred,
            factual_pred=fact_pred,
            sentiment_pred=sentiment_pred,
            type_model=type_model_choice,
            fact_model=fact_model_choice,
            sentiment_model=sentiment_model_choice,
            type_model_label=type_model_label,
            fact_model_label=fact_model_label,
            sentiment_model_label=sentiment_model_label
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)