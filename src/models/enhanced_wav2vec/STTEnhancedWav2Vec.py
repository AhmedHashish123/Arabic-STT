import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torch.nn.functional import log_softmax
import torchaudio
from models.STTBase import STTBase


class CustomSTTModel(nn.Module):
  def __init__(self, wav2vec_model, lstm_hidden_size, lstm_layers, attention_heads):
    super(CustomSTTModel, self).__init__()
    self.wav2vec = wav2vec_model
    for param in self.wav2vec.parameters():
      param.requires_grad = False
    feature_size = self.wav2vec.config.hidden_size
    self.lstm = nn.LSTM(input_size=feature_size,
                        hidden_size=lstm_hidden_size,
                        num_layers=lstm_layers,
                        batch_first=True)
    self.attention = nn.MultiheadAttention(embed_dim=lstm_hidden_size,
                                            num_heads=attention_heads,
                                            batch_first=True)
    self.output_layer = nn.Linear(lstm_hidden_size, wav2vec_model.config.vocab_size)

  def forward(self, input_values, input_lengths):
    self.wav2vec.eval()
    with torch.no_grad():
      wav2vec_output = self.wav2vec(input_values).last_hidden_state
    processed_lengths = wav2vec_output.shape[1]
    processed_lengths = torch.full((wav2vec_output.shape[0],), processed_lengths, dtype=torch.int64)
    sorted_lengths, sorted_indices = input_lengths.sort(descending=True)
    sorted_wav2vec_output = wav2vec_output[sorted_indices]
    packed_input = pack_padded_sequence(sorted_wav2vec_output, processed_lengths.cpu(), batch_first=True)
    if input_values.is_cuda:
      self.lstm.flatten_parameters()
    packed_lstm_output, _ = self.lstm(packed_input)
    lstm_output, _ = pad_packed_sequence(packed_lstm_output, batch_first=True)
    attention_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
    output = self.output_layer(attention_output)
    return output

class STTEnhancedWav2Vec(STTBase):
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained("othrif/wav2vec2-large-xlsr-arabic")
        self.wav2vec_model = Wav2Vec2Model.from_pretrained("othrif/wav2vec2-large-xlsr-arabic")
        self.model = CustomSTTModel(self.wav2vec_model, lstm_hidden_size=128, lstm_layers=2, attention_heads=4)
        self.model.load_state_dict(torch.load("D:\\Ahmed Master's\\Neural Networks\\Project\\Project\\Arabic-STT\\src\\models\\enhanced_wav2vec\\best_model_state_dict.pth"))
        self.model.to(self.device)
        self.model.eval()
        
    def select_outputs(self, outputs, blank_label):
        arg_maxes = torch.argmax(outputs, dim=2)
        # print(arg_maxes.shape)
        decodes = []
        for i in range(arg_maxes.size(0)):
            decode = []
            for j in range(arg_maxes.size(1)):
                if arg_maxes[i][j] != blank_label:
                    if j != 0 and arg_maxes[i][j-1] == arg_maxes[i][j]:
                        continue
                    decode.append(arg_maxes[i][j].item())
            decodes.append(decode)
        return decodes
    
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
        input_values, input_length = self.process_audio_file(audio, self.processor)
        input_values = input_values.to(self.device)
        decoded_text = None
        with torch.no_grad():
            logits = self.model(input_values, input_length)
            log_probs = log_softmax(logits, dim=2)
            decoded_preds = self.select_outputs(log_probs, blank_label=self.processor.tokenizer.pad_token_id)
            decoded_text = [self.processor.decode(pred) for pred in decoded_preds]
        with open("m4.txt", 'w', encoding='utf-8') as file:
            file.write(decoded_text[0])
        return decoded_text[0]