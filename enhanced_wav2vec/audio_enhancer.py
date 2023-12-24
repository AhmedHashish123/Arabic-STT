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

    # Gentle equalization (high-pass filter with a lower cutoff frequency)
    # Lower cutoff frequency to preserve lower frequencies in the voice
    sos = scipy.signal.butter(10, 80, 'hp', fs=sample_rate, output='sos')  # Reduced from 100 to 80 Hz to take into consideration the low noises
    filtered_audio = scipy.signal.sosfilt(sos, reduced_noise_audio)

    # Normalization (increase volume to a target peak level)
    peak_norm_audio = librosa.util.normalize(filtered_audio, norm=np.inf, axis=0)

    # Save the enhanced audio
    sf.write(output_path, peak_norm_audio, sample_rate)

# Usage
input_audio_path = 'r1.wav'  # Replace with your audio file path
output_audio_path = 'r.wav'  # Output file path

enhance_audio(input_audio_path, output_audio_path)
