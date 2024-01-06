import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import numpy as np
import wave
from time import time
from datetime import datetime  
from utils.FileHandler import FileHandler
from models.whisper_fine_tuned.STTWhisper import STTWhisper
FILE_PATH = "data/recordings/"


class AudioRecorderApp:
    def __init__(self, master, filename):
        self.file = FileHandler(filename)
        self.model = STTWhisper()
        self.master = master
        self.master.title("Audio Recorder")
        self.master.protocol("WM_DELETE_WINDOW", self.cleanup_handler)

        # Set window size
        self.master.geometry("400x200")

        # Change background color
        self.master.configure(bg="#f0f0f0")

        # Change font style
        font_style = ("Arial", 14, "bold")

        self.greeting_label = ttk.Label(
            self.master,
            text="Welcome to the Audio Recorder!",
            font=font_style,
            background="#f0f0f0",
        )
        self.greeting_label.pack(pady=10)

        self.record_button = ttk.Button(
            self.master, text="Record", command=self.toggle_record, style="TButton"
        )
        self.record_button.pack(pady=10)

        # Add a label
        self.display_label = ttk.Label(
            self.master, text="Transcription:", font=font_style, background="#f0f0f0"
        )
        self.display_label.pack(pady=5)

        # Add a StringVar to update the label text
        self.additional_text_var = tk.StringVar()
        self.additional_text_var.set("No additional text")  # Initial text
        self.output_label = ttk.Label(self.master, textvariable=self.additional_text_var, font=font_style, background='#f0f0f0')
        self.output_label.pack(pady=5)

        self.is_recording = False
        self.audio_data = []

    def toggle_record(self):
        if not self.is_recording:
            self.start_record()
        else:
            self.stop_record()

    def start_record(self):
        self.is_recording = True
        self.record_button.config(text="Stop Recording")

        def callback(indata, frames, time, status):
            if status:
                print(status, flush=True)
            self.audio_data.append(indata.copy())

        self.stream = sd.InputStream(callback=callback, channels=1, samplerate=16000)
        self.stream.start()

    def stop_record(self):
        self.is_recording = False
        self.record_button.config(text="Record")

        self.stream.stop()

        
        
        date_time = datetime.fromtimestamp(time())
        str_date_time = date_time.strftime("%d-%m-%Y_%H-%M-%S")
        filename = FILE_PATH + str_date_time + ".wav"

        self.save_audio(filename, np.vstack(self.audio_data), sample_rate=16000)
        
        print(f"Audio saved as {filename}")

        transcribed_text = self.model.transcribe(filename)

        parts = transcribed_text.rsplit(' ', 1)

        transcribed_name = parts[0]
        transcribed_mark = parts[1]

        try:
            self.file.update_marks(transcribed_name, transcribed_mark)
        except ValueError as e:
            transcribed_text += "\n" + str(e)

        self.additional_text_var.set(transcribed_text)
        
        self.audio_data = []

    def save_audio(self, filename, audio_data, sample_rate=16000):
        audio_data = np.int16(audio_data * 32767)  # Convert to 16-bit PCM format
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())

    def cleanup_handler(self):
        self.file.write_file()
        
        if self.is_recording:
            self.stop_record()

        # Close the Tkinter window
        self.master.destroy()