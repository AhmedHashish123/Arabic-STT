import torch
from torch import nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Config, Wav2Vec2Model
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import soundfile as sf
import torchaudio

from huggingface_hub import notebook_login

notebook_login()

# Load Common Voice dataset
dataset = load_dataset("mozilla-foundation/common_voice_13_0", "ar", split="validation+train+test")

print(dataset)

# Removing unwanted features (we only want audio and its transcription)
dataset = dataset.remove_columns(["client_id", "path", "up_votes", "down_votes", "age", "gender", "accent", "locale", "segment", "variant"])

print(dataset)

print(dataset[0])

vocabulary = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'ء', 'آ', 'أ', 'إ', 'ؤ', 'ئ', 'ة', 'ى', 'ﻻ', 'ﻷ', 'ﻹ', 'ﻵ',' ', '.']

vocabulary_set = set(vocabulary)  # Convert your existing vocabulary list to a set

# Initialize an empty set for characters from the dataset
dataset_characters = set()

for data_item in dataset:
    sentence = data_item['sentence']
    dataset_characters.update(sentence)

# Perform a union of the dataset characters with the existing vocabulary set
combined_vocabulary = vocabulary_set.union(dataset_characters)

# If you need the combined vocabulary as a list
combined_vocabulary_list = list(combined_vocabulary)

print(len(combined_vocabulary_list))

print(combined_vocabulary_list)

class CustomProcessor:
    def __init__(self, vocab):
        self.vocab = vocab
        self.char_to_index = {char: index for index, char in enumerate(vocab)}

    def text_to_int(self, text):
        return [self.char_to_index[char] for char in text]

    def int_to_text(self, indices):
        return ''.join([self.vocab[index] for index in indices])

object_voc = CustomProcessor(combined_vocabulary_list)

text = "مرحبا"
encoding = object_voc.text_to_int(text)
decoding = object_voc.int_to_text(encoding)
print(encoding)
print(decoding)

import torch
from torch.utils.data import Dataset
import torchaudio

class CommonVoiceDataset(Dataset):
    def __init__(self, dataset, vocab):
        self.dataset = dataset
        self.processor = CustomProcessor(vocab)
        self.spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load and resample audio
        audio_input, sampling_rate = torchaudio.load(self.dataset[idx]["audio"]["path"])
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
common_voice_dataset = CommonVoiceDataset(dataset, combined_vocabulary_list)

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
import torch.nn as nn

class CustomSTTModel2(nn.Module):
    def __init__(self, num_classes):
        super(CustomSTTModel2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gru1 = nn.GRU(input_size=2048, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(input_size=256, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(256, 256)
        self.out = nn.Linear(256, num_classes + 1)  # num_classes includes the Arabic charset + 1 for CTC blank

    def forward(self, x):
        # print("cccccccccccccccccccccccccc")
        # x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)
        # print("dddddddddddddddddddddddddddd")
        # x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)
        # print(x.shape)
        # torch.Size([8, 64, 32, 166]) [Batch Size, Channels, Height(frequency), Width(time)]
        # print("eeeeeeeeeeeeeeeeeeeeeeeeeee")
        x = x.permute(0, 3, 1, 2)  # Rearrange dimensions for GRU; as sequence models expect [Sequence Length (Time Steps),  Batch Size, Feature Size (Number of Features per Time Step)]
        # This often involves flattening the non-time dimensions (like channels and frequency bins) into a single feature dimension
        # print(x.shape)
        # torch.Size([8, 166, 64, 32]) [Batch Size, Width, Channels, Height]
        x = torch.flatten(x, start_dim=2)  # Flatten the convolutional features
        # print("fffffffffffffffffffffffffff")
        # print(x.shape)
        # torch.Size([8, 166, 2048]) [Batch Size, Width, Channels * Height]
        x, _ = self.gru1(x)
        # print(x.shape)
        # print("gggggggggggggggggggggggggggg")
        x = self.dropout(x)

        x, _ = self.gru2(x)
        # print(x.shape)
        # print("hhhhhhhhhhhhhhhhhhhhhhhhhhhh")
        x = self.dropout(x)


        # x = torch.relu(self.fc(x))
        x = self.fc(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        # print(x.shape)
        # print("iiiiiiiiiiiiiiiiiiiiiiiii")
        return x

# Instantiate the model
custom_model_2 = CustomSTTModel2(len(combined_vocabulary_list))

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
train_loader = DataLoader(common_voice_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

best_loss = 10000

def train(model, data_loader, criterion, optimizer, epochs, device):
  global best_loss
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
criterion = nn.CTCLoss(blank=len(combined_vocabulary_list)).to(device)

# Define the Optimizer
optimizer = optim.Adam(custom_model_2.parameters(), lr=0.001)

# Run the training
epochs = 10  # Define the number of epochs
train(custom_model_2, train_loader, criterion, optimizer, epochs, device)