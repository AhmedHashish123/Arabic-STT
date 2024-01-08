import torch
from torch import nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Config, Wav2Vec2Model
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
import soundfile as sf
import torchaudio

# Load Common Voice dataset
common_voice = load_dataset("mozilla-foundation/common_voice_13_0", "ar", split="train+test+validation")

# print(len(common_voice))

# print(common_voice)

# Removing unwanted features (we only want audio path and its transcription)
common_voice = common_voice.remove_columns(["client_id", "audio", "up_votes", "down_votes", "age", "gender", "accent", "locale", "segment", "variant"])

# print(common_voice)

# print(common_voice[0])

fleurs_train = load_dataset("google/fleurs", "ar_eg", split="train")

fleurs_test = load_dataset("google/fleurs", "ar_eg", split="test")

fleurs_val = load_dataset("google/fleurs", "ar_eg", split="validation")

# print(len(fleurs_train))
# print(len(fleurs_test))
# print(len(fleurs_val))

# print(fleurs_train)

def update_audio_path_train(data_item):
    parts = data_item['path'].split('/')
    parts.insert(-1, 'train')
    data_item['path'] = '/'.join(parts)
    data_item['sentence'] = data_item['transcription']
    return data_item
def update_audio_path_test(data_item):
    parts = data_item['path'].split('/')
    parts.insert(-1, 'test')
    data_item['path'] = '/'.join(parts)
    data_item['sentence'] = data_item['transcription']
    return data_item
def update_audio_path_val(data_item):
    parts = data_item['path'].split('/')
    parts.insert(-1, 'dev')
    data_item['path'] = '/'.join(parts)
    data_item['sentence'] = data_item['transcription']
    return data_item

# Apply the transformation
fleurs_train = fleurs_train.map(update_audio_path_train)

# Apply the transformation
fleurs_test = fleurs_test.map(update_audio_path_test)

# Apply the transformation
fleurs_val = fleurs_val.map(update_audio_path_val)

# Removing unwanted features (we only want audio path and its transcription)
fleurs_train = fleurs_train.remove_columns(["id", "num_samples", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"])

fleurs_test = fleurs_test.remove_columns(["id", "num_samples", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"])

fleurs_val = fleurs_val.remove_columns(["id", "num_samples", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"])

# print(fleurs_train[0])
# print(fleurs_test[0])
# print(fleurs_val[0])

combined_dataset = concatenate_datasets([common_voice, fleurs_train, fleurs_test])

# print(len(common_voice))
# print(len(fleurs_train))
# print(len(fleurs_test))
# print(len(fleurs_val))
# print(len(combined_dataset))

# print(combined_dataset)

# print(combined_dataset[0])

vocabulary = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'ء', 'آ', 'أ', 'إ', 'ؤ', 'ئ', 'ة', 'ى', 'ﻻ', 'ﻷ', 'ﻹ', 'ﻵ',' ', '.']

# print(len(vocabulary))

def process_transcriptions(data_item):
    new_sentence = ''.join([char for char in data_item["sentence"] if char in vocabulary])
    data_item['sentence'] = new_sentence
    return data_item

# Apply the transformation
combined_dataset = combined_dataset.map(process_transcriptions)

fleurs_val = fleurs_val.map(process_transcriptions)

# print(combined_dataset[0])

class CustomProcessor:
    def __init__(self, vocab):
        self.vocab = vocab
        self.char_to_index = {char: index for index, char in enumerate(vocab)}

    def text_to_int(self, text):
        return [self.char_to_index[char] for char in text]

    def int_to_text(self, indices):
        return ''.join([self.vocab[index] for index in indices])

object_voc = CustomProcessor(vocabulary)

text = "مرحبا"
encoding = object_voc.text_to_int(text)
decoding = object_voc.int_to_text(encoding)
# print(encoding)
# print(decoding)

import torch
from torch.utils.data import Dataset
import torchaudio

class CustomDataset(Dataset):
    def __init__(self, dataset, vocab):
        self.dataset = dataset
        self.processor = CustomProcessor(vocab)
        self.spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load and resample audio
        audio_input, sampling_rate = torchaudio.load(self.dataset[idx]["path"])
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
            audio_input = resampler(audio_input)

        # Convert to spectrogram
        # spectrogram = self.spectrogram_transform(audio_input).squeeze(0)
        spectrogram = self.spectrogram_transform(audio_input)
        # Normalize spectrogram
        spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()

        # Process text
        sentence = self.dataset[idx]["sentence"]

        labels = self.processor.text_to_int(sentence)
        labels = torch.tensor(labels)

        return spectrogram, labels, spectrogram.shape[-1]

# Instantiate the dataset
custom_data_set = CustomDataset(combined_dataset, vocabulary)
custom_val_data_set = CustomDataset(fleurs_val, vocabulary)

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import math

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

import torch
import torch.nn as nn

class CustomSTTModel2(nn.Module):
    def __init__(self, num_classes):
        super(CustomSTTModel2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # New layer
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # New layer
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Additional layers for skip connections
        # Skip connection convolutions to match channel dimensions
        self.skip_conv2 = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
        self.skip_conv3 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
        self.skip_conv4 = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))

        self.pos_encoder = PositionalEncoding(d_model=8192)  # Set d_model to match the input size of GRU1

        self.gru1 = nn.GRU(input_size=8192, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(input_size=256, hidden_size=256, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(512, 256)
        self.out = nn.Linear(256, num_classes + 1)  # num_classes includes the Arabic charset + 1 for CTC blank

    def forward(self, x):
        # print("cccccccccccccccccccccccccc")
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ELU()(x)
        # x is torch.Size([8, 32, 128, 490])
        x = x + identity  # Element-wise addition (broadcasting works)
        x = self.pool(x)
        # print("dddddddddddddddddddddddddddd")

        identity = self.skip_conv2(x) # Downsampling skip connection
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ELU()(x)
        x = x + identity  # Element-wise addition
        x = self.pool(x)

        # print(x.shape)
        # torch.Size([8, 64, 32, 166]) [Batch Size, Channels, Height(frequency), Width(time)]

        identity = self.skip_conv3(x) # Downsampling skip connection
        x = self.conv3(x)  # Applying new convolutional layer
        x = self.bn3(x)
        x = nn.ELU()(x)
        x = x + identity  # Element-wise addition
        # x = self.pool(x) Removed due to numerical instability
        # print(x.shape)
        # print(identity.shape)
        # torch.Size([8, 128, 32, 166]) [Batch Size, Channels, Height(frequency), Width(time)]

        identity = self.skip_conv4(x) # Downsampling skip connection
        x = self.conv4(x)  # Applying new convolutional layer
        x = self.bn4(x)
        x = nn.ELU()(x)
        x = x + identity  # Element-wise addition
        # x = self.pool(x) Removed due to numerical instability
        # print(x.shape)
        # torch.Size([8, 256, 32, 166]) [Batch Size, Channels, Height(frequency), Width(time)]
        # print("eeeeeeeeeeeeeeeeeeeeeeeeeee")
        x = x.permute(0, 3, 1, 2)  # Rearrange dimensions for GRU; as sequence models expect [Sequence Length (Time Steps),  Batch Size, Feature Size (Number of Features per Time Step)]

        # This often involves flattening the non-time dimensions (like channels and frequency bins) into a single feature dimension
        # print(x.shape)
        # torch.Size([8, 166, 256, 32]) [Batch Size, Width, Channels, Height]
        x = torch.flatten(x, start_dim=2)  # Flatten the convolutional features
        # print("fffffffffffffffffffffffffff")
        # print(x.shape)
        # torch.Size([8, 166, 8192]) [Batch Size, Width, Channels * Height]
        # Apply positional encoding
        x = self.pos_encoder(x)

        # print(x.shape)

        x, _ = self.gru1(x)
        # print(x.shape)
        # print("gggggggggggggggggggggggggggg")
        x = self.dropout(x)

        x, _ = self.gru2(x)
        # print(x.shape)
        # print("hhhhhhhhhhhhhhhhhhhhhhhhhhhh")
        x = self.dropout(x)


        x = self.fc(x)
        x = nn.ELU()(x)
        x = self.dropout(x)
        x = self.out(x)
        # print(x.shape)
        # print("iiiiiiiiiiiiiiiiiiiiiiiii")
        # print(x)
        return x

# Instantiate the model
custom_model_2 = CustomSTTModel2(len(vocabulary))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # Separate the data items
    spectrograms, labels, input_lengths = zip(*batch)

    spectrograms = [s.permute(2, 0, 1) for s in spectrograms]  # Shape: [time_steps, channels, n_mels] since we need to pad the time_steps and pad_sequence pads the first dimension

    # Pad the spectrograms to have the same time length
    # Note: pad_sequence expects a list of tensors, padding them to match the longest tensor
    spectrograms = pad_sequence(spectrograms, batch_first=True, padding_value=0)

    # Permute back to original dimension order after padding
    spectrograms = spectrograms.permute(0, 2, 3, 1)  # Shape: [batch, channels, n_mels, time_steps]

    # Concatenate all labels lengths
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long, device=device)

    # Concatenate all labels (no need for padding)
    labels = torch.cat(labels)

    # Convert input_lengths into a tensor
    input_lengths = torch.tensor(input_lengths)

    return spectrograms, labels, input_lengths, label_lengths

# Create a DataLoader
batch_size = 8
train_loader = DataLoader(custom_data_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(custom_val_data_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def validate(model, data_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # No gradients needed for validation
        for batch_idx, (spectrograms, labels, input_lengths, label_lengths) in enumerate(data_loader):
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            outputs = model(spectrograms)
            output_lengths = torch.full((outputs.shape[0],), outputs.shape[1], dtype=torch.int64)

            log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
            log_probs = log_probs.permute(1, 0, 2).to(device)
            label_lengths = label_lengths.to(device)
            output_lengths = output_lengths.to(device)

            loss = criterion(log_probs, labels, output_lengths, label_lengths)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

best_loss = 10000
counter_inf = 0
def train(model, data_loader, val_loader, criterion, optimizer, epochs, device):
  global best_loss
  global counter_inf
  model.to(device)
  for epoch in range(epochs):
      model.train()  # Set the model to training mode
      total_loss = 0
      for batch_idx, (spectrograms, labels, input_lengths, label_lengths) in enumerate(train_loader):
          spectrograms, labels = spectrograms.to(device), labels.to(device)
          # Forward pass
          outputs = model(spectrograms) # [batch, output sequence length, classes]
          output_lengths = outputs.shape[1]
          output_lengths = torch.full((outputs.shape[0],), output_lengths, dtype=torch.int64)
          # print(outputs)
          log_probs = torch.nn.functional.log_softmax(outputs, dim=2) # CTC expects log softmax probabilities
          # print(log_probs.shape)

          # The output of the network needs to be in the shape (output sequence length, batch, classes)
          log_probs = log_probs.permute(1, 0, 2)
          # print(log_probs.shape)
          log_probs = log_probs.to(device)
          label_lengths = label_lengths.to(device)
          output_lengths = output_lengths.to(device)
          # print(output_lengths)
          # print(log_probs.shape)
          # print(label_lengths)
          # print(labels.shape)
          # Calculate loss
          loss = criterion(log_probs, labels, output_lengths, label_lengths)
          if torch.isinf(loss): # This is due to a few bad anomalies in the dataset. Removing these anomalies would make this condition irrelevant
            # print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
            # print(output_lengths)
            # print(log_probs.shape)
            # print(label_lengths)
            # print(labels.shape)
            # print(outputs)
            # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            # print(log_probs)
            counter_inf += 1
            print("aaaaaaaaaaaaaaaaaaaa")
            continue

          total_loss += loss.item()

          # Backward pass and optimization
          optimizer.zero_grad()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
          optimizer.step()

          # Print progress
          print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item():.4f}")

      # Compute average loss for the epoch
      avg_loss = total_loss / len(data_loader)

      # Perform validation
      avg_val_loss = validate(model, val_loader, criterion, device)
      print("validation loss: ", avg_val_loss)

      # Save the model every two epochs
      if epoch % 2 == 0:
          torch.save(model.state_dict(), f'model_state_dict_epoch_{epoch}.pth')

      # Update the best model if current model is better
      if avg_loss < best_loss:
          best_loss = avg_loss
          torch.save(model.state_dict(), 'best_model_state_dict.pth')

from torch import nn
import torch.optim as optim

# Define the Loss Function (Criterion)
# The CTC loss function expects logits as inputs, so ensure your model outputs logits
criterion = nn.CTCLoss(blank=len(vocabulary)).to(device)

# Define the Optimizer
optimizer = optim.Adam(custom_model_2.parameters(), lr=0.0005)

# Run the training
epochs = 10  # Define the number of epochs
train(custom_model_2, train_loader, validation_loader, criterion, optimizer, epochs, device)

print(counter_inf)