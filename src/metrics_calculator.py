from datasets import load_dataset
from models.whisper_fine_tuned.STTWhisper import STTWhisper
from models.enhanced_wav2vec.STTEnhancedWav2Vec import STTEnhancedWav2Vec
from models.new_architecture.STTNewArchitecture import STTNewArchitecture
from utils.metrics import calculate_wer

whisper_small = STTWhisper("small")
enhanced_wav2vec = STTEnhancedWav2Vec()
new_architecture = STTNewArchitecture()

vocabulary = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'ء', 'آ', 'أ', 'إ', 'ؤ', 'ئ', 'ة', 'ى', 'ﻻ', 'ﻷ', 'ﻹ', 'ﻵ',' ', '.']

fleurs_test = load_dataset("google/fleurs", "ar_eg", split="test")

def update_audio_path_test(data_item):
    parts = data_item['path'].split('\\')
    parts.insert(-1, 'test')
    data_item['path'] = '\\'.join(parts)
    data_item['sentence'] = data_item['transcription']
    return data_item

fleurs_test = fleurs_test.map(update_audio_path_test)

fleurs_test = fleurs_test.remove_columns(["id", "num_samples", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"])

def process_transcriptions(data_item):
  new_sentence = ''.join([char for char in data_item["sentence"] if char in vocabulary])
  data_item["sentence"] = new_sentence
  return data_item

fleurs_test = fleurs_test.map(process_transcriptions)

calculate_wer(fleurs_test, whisper_small, enhanced_wav2vec, new_architecture)