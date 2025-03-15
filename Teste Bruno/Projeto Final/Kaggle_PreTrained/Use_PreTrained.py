# %%
import pickle
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import model_from_json

# %%
# -----------------------------
# Load the pre-trained model
# -----------------------------
# Load the model architecture from JSON file
with open('CNN_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# Load the weights into the model
loaded_model.load_weights('best_model1_weights.h5')
print("Loaded model from disk")

# Compile the model (required before prediction)
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %%
# -----------------------------
# Load the scaler and encoder
# -----------------------------
with open('scaler2.pickle', 'rb') as f:
    scaler = pickle.load(f)
with open('encoder2.pickle', 'rb') as f:
    encoder = pickle.load(f)
print("Loaded scaler and encoder")

# %%
# -----------------------------
# Feature extraction functions
# -----------------------------
def zcr(data, frame_length, hop_length):
    """Compute the zero crossing rate"""
    return np.squeeze(librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length))


def rmse(data, frame_length=2048, hop_length=512):
    """Compute the root mean square energy"""
    return np.squeeze(librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length))


def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    """Extract MFCC features from audio"""
    mfcc_features = librosa.feature.mfcc(data, sr=sr, n_mfcc=30)
    if flatten:
        return np.ravel(mfcc_features.T)
    else:
        return np.squeeze(mfcc_features.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    """Extract and concatenate audio features"""
    features = np.hstack((
        zcr(data, frame_length, hop_length),
        rmse(data, frame_length, hop_length),
        mfcc(data, sr, frame_length, hop_length)
    ))
    return features

def get_predict_feat(path):
    """
    Load an audio file, extract its features, scale and reshape them to be ready for prediction.
    """
    # Load audio file with a fixed duration and offset
    data, sr = librosa.load(path, duration=2.5, offset=0.6)
    features = extract_features(data, sr)
    
    # Reshape to match the model's expected input (1, num_features)
    result = np.array(features).reshape(1, -1)
    
    # Scale the features using the loaded scaler
    result = scaler.transform(result)
    
    # Expand dims to add a channel dimension: (1, num_features, 1)
    result = np.expand_dims(result, axis=2)
    return result

# %%

# -----------------------------
# Prediction function
# -----------------------------
def predict_emotion(audio_path):
    """
    Predict the emotion from an audio file.
    
    Parameters:
        audio_path (str): Path to the audio file.
        
    Returns:
        str: Predicted emotion label.
    """
    features = get_predict_feat(audio_path)
    predictions = loaded_model.predict(features)
    # Inverse transform the one-hot prediction to get the emotion label
    predicted_emotion = encoder.inverse_transform(predictions)
    return predicted_emotion[0][0]




# %%
# -----------------------------
# Example usage
# -----------------------------
if __name__ == '__main__':
    # Replace with the path to your audio file for prediction
    test_audio_path = 'record_out.wav'
    emotion = predict_emotion(test_audio_path)
    print("Predicted emotion:", emotion)


