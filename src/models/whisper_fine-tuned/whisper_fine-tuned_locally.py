from datasets import load_dataset, DatasetDict
common_voice = DatasetDict()


# Combine both training and validation splits into one since Arabic dataset is small
common_voice["train"] = load_dataset("mozilla-foundation/common_voice_13_0", "ar", split="train+validation")
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_13_0", "ar", split="test")


common_voice = common_voice.remove_columns(["client_id", "path", "up_votes", "down_votes", "age", "gender", "accent", "locale", "segment", "variant"])


from transformers import WhisperProcessor
# WhisperProcesor combines both feature extractor and tokenizer
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="Arabic", task="transcribe")


print('A' * 100)


# We need to change the sample rate from 48KHz to 16KHz since this is what whisper expects
from datasets import Audio
# cast_column makes datasets perform the resampling on the fly when the data is loaded
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))


def prepare_dataset(data_item):
    # loading the data item to resample it
    audio = data_item["audio"]
    sentence = data_item["sentence"]
    # compute log-Mel input features from input audio array and add it to our item
    data_item["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"])["input_features"][0]
    # encode target text to label ids and add it to our items
    data_item["labels"] = processor.tokenizer(sentence)["input_ids"]
    # the returned item will only have input_features and labels
    return data_item


# apply prepare_dataset function to all the training data and remove the original columns (audio and sentence)
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)


print('B' * 100)


# creating a class to get the data and batch it
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
@dataclass
class DataCollatorWhisper:
    processor: Any
    # data will be passed to this function
    def __call__(self, data_batch: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply converting them to PyTorch tensors and nothing more
        # no padding will be done since all input_features are padded to 30s and converted to a log-Mel spectrogram of fixed dimension before
        input_features = [{"input_features": feature["input_features"]} for feature in data_batch] # list of features where each element is the dictionary containing the feature vector of a data item from the data batch
        # pad() searches for the longest input features vector and pads the rest to be just like it in length, "pt" means PyTorch which indicates the returned feature as PyTorch tensor
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt") # dictionary containing a list of audio features as PyTorch tensors.
        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in data_batch]
        # pad the labels to max length to make them all have the same length
        # for two audio files with input_id vectors of length 16 and 23, after padding, an attention_mask is created
        # attention_mask will contain two vectors coinciding with the two vectors of input_ids
        # their length is 23 each containing 1s and 0s, 0s at an index means that these elements have been padded at that index
        # so, the first attention_mask vector which corresponds to input_id 16, will have 0s starting from index 16 till 22
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # replace padding with -100 to ignore these tokens when calculating loss according to whisper requirements
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        # if bos token is appended in previous tokenization step
        # remove it token here as it's appended later
        # .all checks if this condition is true for all sequences in the batch
        # .cpu().item() converts the result from a tensor to a boolean to evaluate the if condition
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
    

data_collator = DataCollatorWhisper(processor=processor)


import evaluate
metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # replace -100 with the pad_token_id to allow the decoder to decode them back to strings (so that it doesn't try to decode -100 back to a string)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")


print('C' * 100)


model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


from transformers import Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir="whisper-tiny-ar",
    per_device_train_batch_size=16, # batch size for training per GPU/CPU
    learning_rate=1e-5,
    warmup_steps=500, # linear warmup (from 0 to learning_rate)
    max_steps=4000, # a step a batche of data will be processed the model parameters will be updated based on that batch. 4000 steps will be processed regardless of the number of epochs.
    gradient_checkpointing=True, # saves memory by recomputing some activations during the backward pass instead of storing all the activation values. This takes more time.
    fp16=True, # use 16-bit mixed precision during training instead of 32
    evaluation_strategy="steps", # evaluation is done every eval_steps
    eval_steps=1000,
    per_device_eval_batch_size=8, # batch size for evaluation per GPU/CPU
    predict_with_generate=True, # allows the model to generate entire sequences for evaluation, instead of just single tokens
    generation_max_length=225, # the max number of tokens to be generated during evaluation
    save_steps=1000, # a checkpoint is saved every 1000 steps
    logging_steps=25, # when to receive logs
    load_best_model_at_end=True, # the best model will be loaded at the end of training
    metric_for_best_model="wer", # this metric will be used when comparing different models during training to get the best model
    greater_is_better=False, # it means that a lower value for WER indicates that a model is better
)


from transformers import Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.tokenizer,
)


processor.save_pretrained(training_args.output_dir)


print('D' * 100)


trainer.train()


print('E' * 100)

