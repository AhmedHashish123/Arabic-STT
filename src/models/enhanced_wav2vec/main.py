from utils import preprocess_audio, load_wav2vec_model, wav2vec_feature_extraction

input_audio_path = 'r1.wav'  # Replace with your audio file path

audio = preprocess_audio(input_audio_path)
processor, model = load_wav2vec_model()
features = wav2vec_feature_extraction(audio, processor, model)
print(len(features))
print(features)

