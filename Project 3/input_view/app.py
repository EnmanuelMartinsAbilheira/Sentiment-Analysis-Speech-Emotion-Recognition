from flask import Flask, render_template, request
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Carregar os modelos
with open("models/best_model_Type.pkl", "rb") as f:
    type_model = pickle.load(f)

with open("models/best_model_Factuality.pkl", "rb") as f:
    fact_model = pickle.load(f)

with open("models/best_model_Sentiment.pkl", "rb") as f:
    sentiment_model = pickle.load(f)

# Carregar o vetorizador TF-IDF
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer_fixed = pickle.load(f)

# Carregar o modelo de embeddings (caso necessário)
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Pode mudar conforme o que usou no treino

# Mapeamentos das predições para os rótulos reais
type_mapping = {0: "Affirmation", 1: "Negation"}
fact_subj_mapping = {0: "Factual", 1: "Subjective"}
sentiment_mapping = {0: "Sadness", 1: "Anger", 2: "Neutral", 3: "Happiness", 4: "Euphoria"}

def correct_text(text):
    """
    Simulação de correção gramatical simples.
    Aqui você pode integrar uma API como Grammarly.
    """
    return text.capitalize()

def model_predict(text):
    """
    Faz a predição do texto com os três modelos carregados.
    """
    sentences = [text]

    # Tentar usar embeddings SentenceTransformer
    try:
        X_SenTrans = embedder.encode(sentences).reshape(1, -1)
    except Exception:
        X_SenTrans = None

    # Transformar o texto com TF-IDF
    X_tfidf = vectorizer_fixed.transform(sentences)

    # Fazer previsões
    try:
        type_predictions = type_model.predict(X_SenTrans if X_SenTrans is not None else X_tfidf)
    except Exception:
        type_predictions = type_model.predict(X_tfidf)

    try:
        fact_predictions = fact_model.predict(X_SenTrans if X_SenTrans is not None else X_tfidf)
    except Exception:
        fact_predictions = fact_model.predict(X_tfidf)

    try:
        sentiment_predictions = sentiment_model.predict(X_SenTrans if X_SenTrans is not None else X_tfidf)
    except Exception:
        sentiment_predictions = sentiment_model.predict(X_tfidf)

    # Converter predições para rótulos
    type_label = type_mapping.get(type_predictions[0], "Unknown")
    fact_label = fact_subj_mapping.get(fact_predictions[0], "Unknown")
    sentiment_label = sentiment_mapping.get(sentiment_predictions[0], "Unknown")

    return type_label, fact_label, sentiment_label

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("text")
        corrected = correct_text(text)
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
