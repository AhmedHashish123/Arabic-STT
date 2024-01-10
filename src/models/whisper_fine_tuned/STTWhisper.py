import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from models.STTBase import STTBase



class STTWhisper(STTBase):
    def __init__(self, type="tiny") -> None:
        if(type == "tiny"):
            self.model_dir = "D:\\Ahmed Master's\\Neural Networks\\Project\\Project\\Arabic-STT\\src\\models\\whisper_fine_tuned\\tiny"
        elif(type == "base"):
            self.model_dir = "D:\\Ahmed Master's\\Neural Networks\\Project\\Project\\Arabic-STT\\src\\models\\whisper_fine_tuned\\base"
        elif(type == "small"):
            self.model_dir = "D:\\Ahmed Master's\\Neural Networks\\Project\\Project\\Arabic-STT\\src\\models\\whisper_fine_tuned\\small"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_dir, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_dir, language="Arabic", task="transcribe")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def transcribe(self, audio):
        sample = audio
        result = self.pipe(sample, generate_kwargs={"language": "arabic"})
        with open("m4.txt", 'w', encoding='utf-8') as file:
            file.write(result["text"])
        return result["text"]