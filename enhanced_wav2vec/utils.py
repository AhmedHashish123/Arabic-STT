import librosa
import noisereduce as nr
import numpy as np
import scipy.signal
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

def preprocess_audio(file_path):
    """
    Enhances the audio file by removing noise and rasing the volume
    Takes into account the low noises in the audio file
    Preprocesses the audio file to be used with wav2vec2.0
    """
    # Load the audio file
    audio_data, _ = librosa.load(file_path, sr=16000, mono=True)

    # Noise reduction
    reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=16000)

    # Gentle equalization (high-pass filter with a lower cutoff frequency)
    # Lower cutoff frequency to preserve lower frequencies in the voice
    sos = scipy.signal.butter(10, 80, 'hp', fs=16000, output='sos')  # Reduced from 100 to 80 Hz to take into consideration the low noises
    filtered_audio = scipy.signal.sosfilt(sos, reduced_noise_audio)

    # Normalization (increase volume to a target peak level)
    peak_norm_audio = librosa.util.normalize(filtered_audio, norm=np.inf, axis=0)

    return peak_norm_audio


def load_wav2vec_model():
    """
    Loads the wav2vec2.0 model and processor
    """
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

def wav2vec_feature_extraction(audio_input, processor, model):
    """
    Extracts wav2vec2.0 features from the audio file
    """
    # Process for model input
    input_values = processor(audio_input, return_tensors="pt", sampling_rate = 16000).input_values
    # Get features
    with torch.no_grad():
        features = model(input_values).last_hidden_state

    return features