'''
Esta versão:
    -> corrige os erros todos (ortograficos e gramaticais)

    -> vê a polarity e subjectivity com o textblob

    -> avalia se a frase é negativa ou afirmativa 
            (utiliza a análise do spaCY para procurar palavras de negação como not, n't, never)

NOTA
Ter isto instalado antes:
    -> pip install nltk spacy textblob pyspellchecker gramformer
'''

import nltk
import spacy
import textblob
from textblob import TextBlob
from spellchecker import SpellChecker
from gramformer import Gramformer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class TextAnalyzer:
    def __init__(self):
        self.spell_checker = SpellChecker()
        self.gramformer = Gramformer(models=1, use_gpu=False)
    
    def correct_spelling(self, sentence):
        words = sentence.split() 
        corrected_words = [self.spell_checker.correction(word) or word for word in words] 
        return " ".join(corrected_words) 

    def correct_sentence(self, sentence):
        spelled_corrected = self.correct_spelling(sentence) 
        corrected_sentences = self.gramformer.correct(spelled_corrected, max_candidates=1) 

        return next(iter(corrected_sentences), spelled_corrected)
    
    def analyze_sentiment(self, text):
        """Analyze the sentiment of the text (polarity and subjectivity)."""
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def analyze_sentence_type(self, text):
        """Determine if the sentence is affirmative or negative"""
        doc = nlp(text)
        
        # Check for negation
        has_negation = any(token.dep_ == 'neg' for token in doc)
        
        # Determine sentence type
        if has_negation:
            sentence_type = "negation"
        else:
            sentence_type = "affirmative"
        
        return {
            'sentence_type': sentence_type
        }
    
    def analyze_text(self, text):
        """Complete analysis of the given text."""
        original_text = text
        corrected_text = self.correct_sentence(text)
        
        sentiment = self.analyze_sentiment(corrected_text)
        sentence_params = self.analyze_sentence_type(corrected_text)
        
        result = {
            'original_text': original_text,
            'corrected_text': corrected_text,
            'needs_correction': original_text != corrected_text,
            'polarity': sentiment['polarity'],
            'subjectivity': sentiment['subjectivity'],
            'sentence_type': sentence_params['sentence_type']
        }
        
        return result

def main():
    analyzer = TextAnalyzer()
    
    print("Text Analysis Tool")
    print("------------------")
    print("Enter a sentence to analyze (or 'quit' to exit):")
    
    while True:
        user_input = input("\nYour sentence: ")
        if user_input.lower() == 'quit':
            break
        
        result = analyzer.analyze_text(user_input)
        
        print("\nAnalysis Results:")
        print("-----------------")
        if result['needs_correction']:
            print(f"Corrected: {result['corrected_text']}")
        
        polarity = "positive" if result['polarity'] > 0 else "negative" if result['polarity'] < 0 else "neutral"
        subjectivity = "subjective" if result['subjectivity'] > 0.5 else "objective"
        
        print("Textblob Analysis:")
        print(f"-> Polarity: {polarity} (score: {result['polarity']:.2f})")
        print(f"-> Subjectivity: {subjectivity} (score: {result['subjectivity']:.2f})")
        print("---")
        print(f"Sentence type: {result['sentence_type']}")

if __name__ == "__main__":
    main()
