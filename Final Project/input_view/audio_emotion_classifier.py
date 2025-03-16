import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from IPython.display import display, Audio

class AudioEmotionClassifier:
    """
    A notebook-friendly class for classifying emotions in audio files.
    Designed for interactive use in Jupyter notebooks.
    """
    
    # Class variables
    CLASS_NAMES = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    
    # Important: Fixed figure size for consistent cropping - DO NOT CHANGE
    SPEC_FIG_SIZE = (10, 4)  # This must match the original code's figure size
    SPECTROGRAM_CROP = (90, 37, 805, 342)  # left, top, right, bottom
    IMG_SIZE = (192, 192)
    
    def __init__(self, model_path='best_emotion_model_192x192.h5'):
        """Initialize the classifier with the specified model."""
        self.model = tf.keras.models.load_model(model_path)
        print(f"✓ Model loaded successfully from {model_path}")
    
    def load_audio(self, audio_path):
        """Load an audio file and return it with sample rate."""
        y, sr = librosa.load(audio_path, sr=None)
        print(f"✓ Loaded audio: {os.path.basename(audio_path)} ({len(y)/sr:.2f} seconds)")
        # Display an audio player widget in the notebook
        display(Audio(y, rate=sr))
        return y, sr
    
    def create_spectrogram(self, y=None, sr=None, audio_path=None, save_path=None, display_size=(12, 5)):
        """
        Create and display a mel-spectrogram from audio data or file.
        
        Args:
            y: Audio time series (if already loaded)
            sr: Sampling rate (if y is provided)
            audio_path: Path to audio file (alternative to y, sr)
            save_path: Path to save the spectrogram image (optional)
            display_size: Size for display only (does not affect saved image)
        
        Returns:
            Tuple of (spectrogram data, save_path if saved)
        """
        # Load audio if not provided
        if y is None:
            if audio_path is None:
                raise ValueError("Either audio data (y, sr) or audio_path must be provided")
            y, sr = self.load_audio(audio_path)
            
        # Generate default save path if needed
        if save_path is None and audio_path is not None:
            save_path = f"{os.path.splitext(audio_path)[0]}_mel.png"
            
        # Compute the mel-spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # For saving: use the FIXED figure size that matches the original cropping dimensions
        if save_path:
            plt.figure(figsize=self.SPEC_FIG_SIZE)
            librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
            plt.title(f"Mel-Spectrogram")
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()  # Close the saving figure
            
            # Crop the image
            with Image.open(save_path) as img:
                cropped_img = img.crop(self.SPECTROGRAM_CROP)
                cropped_img.save(save_path)
            print(f"✓ Saved spectrogram to {save_path}")
        
        # For display: use the requested display size
        plt.figure(figsize=display_size)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        title = f"Mel-Spectrogram"
        if audio_path:
            title += f": {os.path.basename(audio_path)}"
        plt.title(title)
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()
        
        return S_dB, save_path if save_path else None
    
    def predict(self, audio_path=None, spectrogram_path=None, y=None, sr=None, display_size=(10, 4)):
        """
        Predict emotion from audio data.
        
        Args:
            audio_path: Path to audio file
            spectrogram_path: Path to pre-generated spectrogram
            y: Audio time series (if already loaded)
            sr: Sampling rate (if y is provided)
            display_size: Size for display plots (does not affect prediction)
            
        Returns:
            Dictionary with prediction results
        """
        # Generate spectrogram if needed
        if spectrogram_path is None:
            if audio_path is not None:
                _, spectrogram_path = self.create_spectrogram(
                    audio_path=audio_path, 
                    display_size=display_size
                )
            elif y is not None and sr is not None:
                _, spectrogram_path = self.create_spectrogram(
                    y=y, 
                    sr=sr, 
                    display_size=display_size
                )
            else:
                raise ValueError("Either audio_path, spectrogram_path, or audio data (y, sr) must be provided")
        
        # Preprocess and predict
        input_image = self._preprocess_image(spectrogram_path)
        predictions = self.model.predict(input_image, verbose=0)
        
        # Determine results
        predicted_class_idx = np.argmax(predictions[0])
        predicted_emotion = self.CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        # Create result dictionary with all probabilities
        probabilities = {
            emotion: float(prob) 
            for emotion, prob in zip(self.CLASS_NAMES, predictions[0])
        }
        
        # Display results in notebook-friendly format
        print(f"\n🎭 Predicted Emotion: {predicted_emotion} ({confidence:.2%} confidence)")
        
        # Create bar chart of probabilities
        plt.figure(figsize=display_size)
        colors = ['#ff9999' if emotion == predicted_emotion else '#9999ff' for emotion in self.CLASS_NAMES]
        bars = plt.bar(self.CLASS_NAMES, list(probabilities.values()), color=colors)
        plt.title('Emotion Prediction Probabilities')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        
        # Place text inside the bars only if probability > 50%
        for i, bar in enumerate(bars):
            height = bar.get_height()
            prob_value = list(probabilities.values())[i]
            
            if prob_value > 0.5:  # Only show label if probability > 50%
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    height/2,  # Position text in middle of bar
                    f"{prob_value:.2%}",
                    ha='center',
                    va='center',
                    color='black',
                    fontweight='bold'
                )
        
        plt.tight_layout()
        plt.show()
        
        return {
            'emotion': predicted_emotion,
            'confidence': confidence,
            'probabilities': probabilities,
            'spectrogram_path': spectrogram_path
        }
    
    def _preprocess_image(self, image_path):
        """Preprocess an image for the model (internal method)."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.IMG_SIZE)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        img = np.expand_dims(img, axis=0)    # Add batch dimension
        return img

# Example usage in notebook cells:

# Cell 1: Import and initialize
# classifier = AudioEmotionClassifier()

# Cell 2: Analyze a file
# result = classifier.predict(audio_path="happy.wav")