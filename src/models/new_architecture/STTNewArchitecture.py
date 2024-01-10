import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torch.nn.functional import log_softmax
import torchaudio
from models.STTBase import STTBase
import math

class CustomProcessor:
  def __init__(self, vocab):
    self.vocab = vocab
    self.char_to_index = {char: index for index, char in enumerate(vocab)}

  def text_to_int(self, text):
    return [self.char_to_index[char] for char in text]

  def int_to_text(self, indices):
    return ''.join([self.vocab[index] for index in indices])

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout=0.1, max_len=5000):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:x.size(0)]
    return self.dropout(x)
  
class CustomSTTModel2(nn.Module):
  def __init__(self, num_classes):
    super(CustomSTTModel2, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.bn2 = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.bn3 = nn.BatchNorm2d(128)
    self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.bn4 = nn.BatchNorm2d(256)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.skip_conv2 = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
    self.skip_conv3 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
    self.skip_conv4 = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
    self.pos_encoder = PositionalEncoding(d_model=8192)
    self.gru1 = nn.GRU(input_size=8192, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
    self.gru2 = nn.GRU(input_size=256, hidden_size=256, num_layers=1, bidirectional=True, batch_first=True)
    self.dropout = nn.Dropout(0.5)
    self.fc = nn.Linear(512, 256)
    self.out = nn.Linear(256, num_classes + 1)
  def forward(self, x):
    skip_connection = x
    x = self.conv1(x)
    x = self.bn1(x)
    x = nn.ELU()(x)
    x = x + skip_connection
    x = self.pool(x)
    skip_connection = self.skip_conv2(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = nn.ELU()(x)
    x = x + skip_connection
    x = self.pool(x)
    skip_connection = self.skip_conv3(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = nn.ELU()(x)
    x = x + skip_connection
    skip_connection = self.skip_conv4(x)
    x = self.conv4(x)
    x = self.bn4(x)
    x = nn.ELU()(x)
    x = x + skip_connection
    x = x.permute(0, 3, 1, 2)
    x = torch.flatten(x, start_dim=2)
    x = self.pos_encoder(x)
    x, _ = self.gru1(x)
    x = self.dropout(x)
    x, _ = self.gru2(x)
    x = self.dropout(x)
    x = self.fc(x)
    x = nn.ELU()(x)
    x = self.dropout(x)
    x = self.out(x)
    return x

class STTNewArchitecture(STTBase):
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocabulary = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'ء', 'آ', 'أ', 'إ', 'ؤ', 'ئ', 'ة', 'ى', 'ﻻ', 'ﻷ', 'ﻹ', 'ﻵ',' ', '.']
        self.spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
        self.model = CustomSTTModel2(len(self.vocabulary))
        self.model.load_state_dict(torch.load("D:\\Ahmed Master's\\Neural Networks\\Project\\Project\\Arabic-STT\\src\\models\\new_architecture\\best_model_state_dict.pth"))
        self.model.to(self.device)
        self.model.eval()
        self.object_voc = CustomProcessor(self.vocabulary)
        
    def transform_audio_to_spectrogram(self, audio_path, transform):
      waveform, sample_rate = torchaudio.load(audio_path)
      if waveform.shape[0] == 2:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
      if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
      spectrogram = transform(waveform)
      spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()
      return spectrogram
    
    def process_audio_file(self, file_path, processor, target_sample_rate=16000):
        audio_input, sampling_rate = torchaudio.load(file_path)
        if audio_input.shape[0] == 2:
            audio_input = torch.mean(audio_input, dim=0, keepdim=True)
        if sampling_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=target_sample_rate)
            audio_input = resampler(audio_input)
        input_values = processor(audio_input, sampling_rate=target_sample_rate, return_tensors="pt").input_values
        input_values = input_values.squeeze(0)
        input_length = input_values.shape[1]
        input_length = torch.tensor([input_length], dtype=torch.long)
        return input_values, input_length
    
    def transcribe(self, audio):
        spectrogram = self.transform_audio_to_spectrogram(audio, self.spectrogram_transform)
        spectrogram = spectrogram.unsqueeze(0).to(self.device)
        with torch.no_grad():
          outputs = self.model(spectrogram)
          outputs = torch.nn.functional.log_softmax(outputs, dim=2)
          outputs = outputs.permute(1, 0, 2)
        decoded_preds = torch.argmax(outputs, dim=2)
        decoded_preds = decoded_preds.transpose(0, 1)
        decoded_preds_list = decoded_preds.flatten().tolist()
        pred_text = self.object_voc.int_to_text([i for i in decoded_preds_list if i < len(self.object_voc.vocab)])
        with open("m4.txt", 'w', encoding='utf-8') as file:
            file.write(pred_text)
        return pred_text