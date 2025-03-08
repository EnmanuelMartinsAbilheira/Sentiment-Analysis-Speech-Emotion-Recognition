from flask import Flask, render_template, request
import random

app = Flask(__name__)

def correct_text(text):
    """
    Função de simulação para correção gramatical.
    Aqui você pode integrar uma API ou biblioteca real para correção.
    """
    # Exemplo simples: apenas coloca a primeira letra em maiúscula
    corrected = text.capitalize()
    return corrected

def model_predict(text):
    """
    Função simulada para realizar a predição com um modelo pré-treinado.
    Gera valores aleatórios para ilustrar as porcentagens.
    """
    # Predição para Type: Affirmation e Negation
    type_affirmation = random.randint(50, 100) 
    type_negation = 100 - type_affirmation

    # Predição para Factual / Subjective
    factual = random.randint(50, 100)
    subjective = 100 - factual

    # Predição para Sentiment: totalizando 100%
    sentiment_neutral = random.randint(0, 100)
    sentiment_anger = random.randint(0, 100 - sentiment_neutral)
    sentiment_sadness = random.randint(0, 100 - sentiment_neutral - sentiment_anger)
    sentiment_happiness = random.randint(0, 100 - sentiment_neutral - sentiment_anger - sentiment_sadness)
    sentiment_euphoria = 100 - (sentiment_neutral + sentiment_anger + sentiment_sadness + sentiment_happiness)

    type_pred = {"Affirmation": type_affirmation, "Negation": type_negation}
    factual_pred = {"Factual": factual, "Subjective": subjective}
    sentiment_pred = {
        "Neutral": sentiment_neutral,
        "Anger": sentiment_anger,
        "Sadness": sentiment_sadness,
        "Happiness": sentiment_happiness,
        "Euphoria": sentiment_euphoria
    }
    
    return type_pred, factual_pred, sentiment_pred

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("text")
        corrected = correct_text(text)
        type_pred, factual_pred, sentiment_pred = model_predict(text)
        return render_template("result.html", 
                               original_text=text, 
                               corrected_text=corrected, 
                               type_pred=type_pred, 
                               factual_pred=factual_pred, 
                               sentiment_pred=sentiment_pred)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
