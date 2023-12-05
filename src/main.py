import tkinter as tk
from app.AudioRecorderApp import AudioRecorderApp

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioRecorderApp(root, "data/test.xls")
    root.mainloop()
