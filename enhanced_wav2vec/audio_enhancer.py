import librosa
import noisereduce as nr
import numpy as np
import scipy.signal
import soundfile as sf

def enhance_audio(file_path, output_path):
    # Load the audio file
    audio_data, sample_rate = librosa.load(file_path, sr=None)

    # Noise reduction
    reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=sample_rate)

    # Simple equalization (high-pass filter to remove low-frequency noise)
    sos = scipy.signal.butter(10, 100, 'hp', fs=sample_rate, output='sos')
    filtered_audio = scipy.signal.sosfilt(sos, reduced_noise_audio)

    # Save the enhanced audio
    sf.write(output_path, reduced_noise_audio, sample_rate)

# Usage
input_audio_path = 'r1.wav'  # Replace with your audio file path
output_audio_path = 'r_new.wav'  # Output file path

enhance_audio(input_audio_path, output_audio_path)
