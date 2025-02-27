'''
Esta versão:
    -> corrige erros ortográficos e gramaticais
        Exemplo:
            Input: "I is testng grammar tool using python. It does not costt anythng."
            Output: "I am testing a grammar tool using Python. It does not cost anything."


NOTA:            
Ter isto instalado antes:

    -> pip install -U git+https://github.com/PrithivirajDamodaran/Gramformer.git
    -> pip install torch
    -> pip install spacy
    -> python -m spacy download en
'''

from spellchecker import SpellChecker
from gramformer import Gramformer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class TextCorrector:
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


def main():
    corrector = TextCorrector()  
    
    print("------------------------")
    print("Enter a sentence (or 'quit' to exit):")

    while True:
        user_input = input("\nYour sentence: ")
        if user_input.lower() == 'quit':
            break

        corrected_text = corrector.correct_sentence(user_input) 

        print("\nCorrected sentence:")
        print(corrected_text) 

if __name__ == "__main__":
    main()