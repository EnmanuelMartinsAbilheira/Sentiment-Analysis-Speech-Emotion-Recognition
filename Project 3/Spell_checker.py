'''
Ter isto instalado antes:
pip install pyspellchecker
'''

from spellchecker import SpellChecker

class TextCorrector:
    def __init__(self):
        self.spell_checker = SpellChecker()
    
    def correct_sentence(self, text):
        words = text.split()
        corrected_words = []
        
        for word in words:
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word and self.spell_checker.correction(clean_word) != clean_word:
                corrected = word.replace(clean_word, self.spell_checker.correction(clean_word))
                corrected_words.append(corrected)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)

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