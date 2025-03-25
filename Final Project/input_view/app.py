from flask import Flask, render_template, request
import sqlite3
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from spellchecker import SpellChecker
from gramformer import Gramformer
import spacy
from transformers import pipeline
import matplotlib.pyplot as plt
import io
import base64
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import sounddevice as sd
import wave
from audio_emotion_classifier import AudioEmotionClassifier
import whisper

app = Flask(__name__)

"""Here we load the custom models"""
# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models and vectorizer with absolute paths
with open(os.path.join(BASE_DIR, "models", "best_model_Type.pkl"), "rb") as f:
    type_model = pickle.load(f)

with open(os.path.join(BASE_DIR, "models", "best_model_Factuality.pkl"), "rb") as f:
    fact_model = pickle.load(f)

with open(os.path.join(BASE_DIR, "models", "best_model_Sentiment.pkl"), "rb") as f:
    sentiment_model = pickle.load(f)

with open(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"), "rb") as f:
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

class OptimizedT5Corrector:
    def __init__(self, debug=False):
        # Load model and tokenizer directly
        self.model_name = "prithivida/grammar_error_correcter_v1"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        
        # Optimize model for inference
        self.model.eval()
        
        # Use torch.compile for PyTorch 2.0+ (significant speedup)
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                if debug: print("Successfully applied torch.compile optimization")
            except Exception as e:
                if debug: print(f"Could not apply torch.compile: {e}")
        
        # Optimize memory usage
        self.model.config.use_cache = True
        
    def correct(self, sentence, max_length=128):
        # Apply inference optimizations
        with torch.inference_mode():
            # Prepare input - the "gec:" prefix is important for the model
            input_text = f"gec: {sentence}"
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
            
            # Optimize generation parameters for speed
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                num_beams=2,  # Reduced from 5 for speed
                early_stopping=True,
                use_cache=True  # Enable KV caching for faster generation
            )
            
            # Decode output
            corrected_sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return corrected_sentence

def correct_spelling(sentence):
    spell_checker = SpellChecker()
    words = sentence.split() 
    corrected_words = [spell_checker.correction(word) or word for word in words] 
    result = " ".join(corrected_words)
    return result

# Initialize the optimized corrector globally for reuse
_optimized_corrector = None
def correct_sentence(sentence):
    global _optimized_corrector
    
    # Initialize the optimized corrector if it hasn't been already
    if _optimized_corrector is None:
        _optimized_corrector = OptimizedT5Corrector(debug=False)
    
    # First correct spelling
    spelled_corrected = correct_spelling(sentence)
    
    # Use the optimized corrector
    result = _optimized_corrector.correct(spelled_corrected)
    
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

# Database setup
def setup_database():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY,
            original_sentence TEXT,
            corrected_sentence TEXT,
            type_prediction TEXT,
            factuality_prediction TEXT,
            sentiment_prediction TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(original_sentence, corrected_sentence, type_prediction, factuality_prediction, sentiment_prediction):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (original_sentence, corrected_sentence, type_prediction, factuality_prediction, sentiment_prediction)
        VALUES (?, ?, ?, ?, ?)
    ''', (original_sentence, corrected_sentence, type_prediction, factuality_prediction, sentiment_prediction))
    conn.commit()
    conn.close()

def generate_plot():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('SELECT type_prediction, COUNT(*) FROM predictions GROUP BY type_prediction')
    type_data = c.fetchall()
    c.execute('SELECT factuality_prediction, COUNT(*) FROM predictions GROUP BY factuality_prediction')
    factuality_data = c.fetchall()
    c.execute('SELECT sentiment_prediction, COUNT(*) FROM predictions GROUP BY sentiment_prediction')
    sentiment_data = c.fetchall()
    conn.close()

    # Generate bar and pie plots for type predictions
    type_labels, type_counts = zip(*type_data)
    plt.figure(figsize=(10, 6))
    plt.bar(type_labels, type_counts, color='skyblue')
    plt.title('Type Predictions Count')
    plt.xlabel('Type Prediction')
    plt.ylabel('Count')
    type_bar_plot = io.BytesIO()
    plt.savefig(type_bar_plot, format='png')
    type_bar_plot.seek(0)
    type_bar_plot_url = base64.b64encode(type_bar_plot.getvalue()).decode()

    plt.figure(figsize=(6, 6))
    plt.pie(type_counts, labels=type_labels, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
    plt.title('Type Predictions Distribution')
    type_pie_plot = io.BytesIO()
    plt.savefig(type_pie_plot, format='png')
    type_pie_plot.seek(0)
    type_pie_plot_url = base64.b64encode(type_pie_plot.getvalue()).decode()

    # Generate bar and pie plots for factuality predictions
    factuality_labels, factuality_counts = zip(*factuality_data)
    plt.figure(figsize=(10, 6))
    plt.bar(factuality_labels, factuality_counts, color='lightgreen')
    plt.title('Factuality Predictions Count')
    plt.xlabel('Factuality Prediction')
    plt.ylabel('Count')
    factuality_bar_plot = io.BytesIO()
    plt.savefig(factuality_bar_plot, format='png')
    factuality_bar_plot.seek(0)
    factuality_bar_plot_url = base64.b64encode(factuality_bar_plot.getvalue()).decode()

    plt.figure(figsize=(6, 6))
    plt.pie(factuality_counts, labels=factuality_labels, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
    plt.title('Factuality Predictions Distribution')
    factuality_pie_plot = io.BytesIO()
    plt.savefig(factuality_pie_plot, format='png')
    factuality_pie_plot.seek(0)
    factuality_pie_plot_url = base64.b64encode(factuality_pie_plot.getvalue()).decode()

    # Generate bar and pie plots for sentiment predictions
    sentiment_labels, sentiment_counts = zip(*sentiment_data)
    plt.figure(figsize=(10, 6))
    plt.bar(sentiment_labels, sentiment_counts, color='lightcoral')
    plt.title('Sentiment Predictions Count')
    plt.xlabel('Sentiment Prediction')
    plt.ylabel('Count')
    sentiment_bar_plot = io.BytesIO()
    plt.savefig(sentiment_bar_plot, format='png')
    sentiment_bar_plot.seek(0)
    sentiment_bar_plot_url = base64.b64encode(sentiment_bar_plot.getvalue()).decode()

    plt.figure(figsize=(6, 6))
    plt.pie(sentiment_counts, labels=sentiment_labels, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
    plt.title('Sentiment Predictions Distribution')
    sentiment_pie_plot = io.BytesIO()
    plt.savefig(sentiment_pie_plot, format='png')
    sentiment_pie_plot.seek(0)
    sentiment_pie_plot_url = base64.b64encode(sentiment_pie_plot.getvalue()).decode()

    return type_bar_plot_url, type_pie_plot_url, factuality_bar_plot_url, factuality_pie_plot_url, sentiment_bar_plot_url, sentiment_pie_plot_url

# Final Project
def voice_to_text_prediction(duration):
    """Graba audio, transcrribe a texto y clasifica emociones usando AudioEmotionClassifier"""


    # Configuração da gravação
    SAMPLE_RATE = 44100
    CHANNELS = 1

    def record_audio(filename="static/output.wav", duration=duration):
        print(f"🎤 Gravando... ({duration} segundos)")
        audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16)
        sd.wait()
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())
        print("⏹️ Gravação concluída.")
        return filename

    def transcribe_audio(audio_path):
        """Transcribe audio to text using Whisper"""
        try:
            print("📝 Transcribing audio...")
            model = whisper.load_model("small")
            result = model.transcribe(audio_path, language="en")
            return result["text"]
        except FileNotFoundError:
            print("Error: FFmpeg not found. Please install FFmpeg first.")
            return "Error: Could not transcribe audio. FFmpeg is not installed."
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            return f"Error: Could not transcribe audio. {str(e)}"

    def get_plot_base64(plt_obj):
        img = io.BytesIO()
        plt_obj.savefig(img, format='png')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()

    # Grabar audio
    audio_file = record_audio(duration=duration)
    
    # Transcribir audio
    transcribed_text = transcribe_audio(audio_file)

    # Inicializar y usar AudioEmotionClassifier
    classifier = AudioEmotionClassifier(model_path='models/best_emotion_model_192x192.h5')
    result = classifier.predict(audio_path=audio_file)

    # Generar y capturar el espectrograma como base64
    spectrogram_base64, S_dB, save_path = classifier.create_spectrogram(
        audio_path=audio_file, 
        display_size=(10, 4)
    )
    spectrogram_plot_url = get_plot_base64(plt.gcf())
    plt.close()

    # Generar y capturar el gráfico de probabilidades como base64
    plt.figure(figsize=(10, 4))
    colors = ['#ff9999' if emotion == result['emotion'] else '#9999ff' for emotion in classifier.CLASS_NAMES]
    bars = plt.bar(classifier.CLASS_NAMES, list(result['probabilities'].values()), color=colors)
    plt.title('Emotion Prediction Probabilities')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        prob_value = list(result['probabilities'].values())[i]
        if prob_value > 0.5:
            plt.text(bar.get_x() + bar.get_width()/2, height/2, f"{prob_value:.2%}", ha='center', va='center', color='black', fontweight='bold')
    plt.tight_layout()
    probability_plot_url = get_plot_base64(plt.gcf())
    plt.close()

    # Return the spectrogram URL along with other results
    return {
        'transcribed_text': transcribed_text,
        'emotion': result['emotion'],
        'confidence': result['confidence'],
        'probability_plot_url': probability_plot_url,
        'spectrogram_plot_url': spectrogram_plot_url
    }












def spectogram_voice_sentiment_analytic(duration=6):
    """Graba audio y clasifica emociones usando AudioEmotionClassifier"""
    import os
    import sounddevice as sd
    import numpy as np
    import wave
    from audio_emotion_classifier import AudioEmotionClassifier
    import matplotlib.pyplot as plt
    import io
    import base64

    # Configuración de la grabación
    SAMPLE_RATE = 44100
    CHANNELS = 1

    def record_audio(filename="static/output.wav", duration=duration, samplerate=SAMPLE_RATE):
        print("🎙️ Gravando... Fale algo!")
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
        sd.wait()
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_data.tobytes())
        print(f"✅ Áudio salvo como {filename}")
        return filename

    def get_plot_base64(plt_obj):
        img = io.BytesIO()
        plt_obj.savefig(img, format='png')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()

    # Grabar audio
    audio_file = record_audio(duration=duration)

    # Inicializar y usar AudioEmotionClassifier
    classifier = AudioEmotionClassifier(model_path='models/best_emotion_model_192x192.h5')
    result = classifier.predict(audio_path=audio_file)

    # Generar y capturar el espectrograma como base64
    # ...existing code...

    # Update this section to correctly unpack three values
    spectrogram_base64, S_dB, save_path = classifier.create_spectrogram(
        audio_path=audio_file, 
        display_size=(10, 4)
    )
    spectrogram_plot_url = get_plot_base64(plt.gcf())
    plt.close()

    # ...rest of the function...

    # Generar y capturar el gráfico de probabilidades como base64
    plt.figure(figsize=(10, 4))
    colors = ['#ff9999' if emotion == result['emotion'] else '#9999ff' for emotion in classifier.CLASS_NAMES]
    bars = plt.bar(classifier.CLASS_NAMES, list(result['probabilities'].values()), color=colors)
    plt.title('Emotion Prediction Probabilities')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        prob_value = list(result['probabilities'].values())[i]
        if prob_value > 0.5:
            plt.text(bar.get_x() + bar.get_width()/2, height/2, f"{prob_value:.2%}", ha='center', va='center', color='black', fontweight='bold')
    plt.tight_layout()
    probability_plot_url = get_plot_base64(plt.gcf())
    plt.close()

    return {
        'emotion': result['emotion'],
        'confidence': result['confidence'],
        'probability_plot_url': probability_plot_url,
        'spectrogram_plot_url': spectrogram_plot_url
    }



@app.route("/spectogram-voice-sentiment", methods=["GET", "POST"])
def spectogram_voice_sentiment():
    if request.method == "POST":
        # Obtener la duración del input si se proporciona, o usar un valor predeterminado
        duration = request.form.get("duration", 6)
        try:
            duration = float(duration)
            if duration <= 0:
                duration = 6
        except ValueError:
            duration = 6

        # Procesar el audio y clasificar emociones
        result = spectogram_voice_sentiment_analytic(duration)

        return render_template(
            "spectogram_result.html",
            emotion=result['emotion'],
            confidence=result['confidence'],
            probability_plot_url=result['probability_plot_url'],
            spectrogram_plot_url=result['spectrogram_plot_url']
        )

    return render_template("index.html")











# Route to display predictions
@app.route("/predictions")
def predictions():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('SELECT * FROM predictions')
    rows = c.fetchall()
    conn.close()
    
    type_bar_plot_url, type_pie_plot_url, factuality_bar_plot_url, factuality_pie_plot_url, sentiment_bar_plot_url, sentiment_pie_plot_url = generate_plot()
    
    return render_template('predictions.html', rows=rows, type_bar_plot_url=type_bar_plot_url, type_pie_plot_url=type_pie_plot_url, factuality_bar_plot_url=factuality_bar_plot_url, factuality_pie_plot_url=factuality_pie_plot_url, sentiment_bar_plot_url=sentiment_bar_plot_url, sentiment_pie_plot_url=sentiment_pie_plot_url)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        
        # Verificar si se envió un formulario de texto o una solicitud de voz
        if "text" in request.form:
            text = request.form.get("text")
        else:
            # Obtener la duración del input si se proporciona, o usar un valor predeterminado
            duration = request.form.get("duration", 5)
            try:
                duration = float(duration)
                if duration <= 0:
                    duration = 10
            except ValueError:
                duration = 10
            result = voice_to_text_prediction(duration)
            text = result['transcribed_text']
            
        model_type = request.form.get("model_type", "custom")  # Get the selected model type
        
        # Correct spelling and grammar
        corrected = correct_sentence(text)
        
        # Initialize model choices based on the selected model_type
        type_model_choice = request.form.get("type_model", model_type)
        fact_model_choice = request.form.get("fact_model", model_type)
        sentiment_model_choice = request.form.get("sentiment_model", model_type)
        
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

        # Save predictions to SQLite
        save_prediction(text, corrected, type_pred, fact_pred, sentiment_pred)

        # Si es una grabación de voz, incluir los resultados de emoción
        if "voice" in request.form:
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
                sentiment_model_label=sentiment_model_label,
                emotion=result['emotion'],
                confidence=result['confidence'],
                probability_plot_url=result['probability_plot_url'],
                spectrogram_plot_url=result['spectrogram_plot_url']
            )
        else:
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
    setup_database()
    app.run(debug=True)
