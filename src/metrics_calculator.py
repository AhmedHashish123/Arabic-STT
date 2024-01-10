from datasets import load_dataset
from models.whisper_fine_tuned.STTWhisper import STTWhisper
from models.enhanced_wav2vec.STTEnhancedWav2Vec import STTEnhancedWav2Vec
from models.new_architecture.STTNewArchitecture import STTNewArchitecture
from utils.metrics import calculate_wer

whisper_tiny = STTWhisper("tiny")
whisper_base = STTWhisper("base")
whisper_small = STTWhisper("small")
enhanced_wav2vec = STTEnhancedWav2Vec()
new_architecture = STTNewArchitecture()

vocabulary = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'ء', 'آ', 'أ', 'إ', 'ؤ', 'ئ', 'ة', 'ى', 'ﻻ', 'ﻷ', 'ﻹ', 'ﻵ',' ', '.']

common_voice = load_dataset("mozilla-foundation/common_voice_13_0", "ar", split="test")

common_voice = common_voice.remove_columns(["client_id", "audio", "up_votes", "down_votes", "age", "gender", "accent", "locale", "segment", "variant"])

def process_transcriptions(data_item):
  new_sentence = ''.join([char for char in data_item["sentence"] if char in vocabulary])
  data_item["sentence"] = new_sentence
  return data_item

common_voice = common_voice.map(process_transcriptions)

calculate_wer(common_voice, whisper_tiny, whisper_base, whisper_small, enhanced_wav2vec, new_architecture)