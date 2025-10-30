import os
import sounddevice as sd
import numpy as np
import openai
import tempfile
import scipy.io.wavfile as wav
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import threading
import queue
from pynput import keyboard

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

samplerate = 44100
chunk_duration = 0.5   # seconds per audio chunk sent to queue
accumulate_time = 15.0 # seconds before sending to Whisper
q = queue.Queue()
running = False
exit_flag = False

def record_audio():
    """Continuously record audio chunks in real time while running."""
    global running, exit_flag
    print("Press 's' to start/stop recording, 'q' to quit.")

    buffer = []
    frames_per_chunk = int(chunk_duration * samplerate)
    accumulated_samples = int(accumulate_time * samplerate)

    def callback(indata, frames, time, status):
        if running:
            buffer.append(indata.copy())
            total_samples = sum(len(b) for b in buffer)
            if total_samples >= accumulated_samples:
                combined = np.concatenate(buffer, axis=0)
                q.put(combined)
                buffer.clear()

    with sd.InputStream(samplerate=samplerate, channels=1, dtype=np.int16, callback=callback):
        while not exit_flag:
            sd.sleep(100)
    # Flush any remaining data
    if buffer:
        combined = np.concatenate(buffer, axis=0)
        q.put(combined)
    q.put(None)

def transcribe_worker():
    """Pull audio chunks and send to Whisper."""
    downloads_path = Path.home() / "Downloads"
    transcript_file = downloads_path / f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(transcript_file, "a") as out:
        while True:
            audio = q.get()
            if audio is None:
                break
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                wav.write(tmpfile.name, samplerate, audio)
                audio_path = tmpfile.name
            try:
                with open(audio_path, "rb") as f:
                    transcript = openai.audio.transcriptions.create(
                        model="whisper-1", 
                        file=f
                    )
                text = transcript.text.strip()
                if text:
                    print("> " + text)
                    out.write(text + "\n")
                    out.flush()
            except Exception as e:
                print("Error:", e)

def on_press(key):
    """Handle key events."""
    global running, exit_flag
    try:
        if key.char == "s":
            running = not running
            print("Recording started." if running else "Recording stopped.")
        elif key.char == "q":
            exit_flag = True
            return False  # stop listener
    except AttributeError:
        pass

# --- Threads ---
rec_thread = threading.Thread(target=record_audio, daemon=True)
trans_thread = threading.Thread(target=transcribe_worker, daemon=True)
rec_thread.start()
trans_thread.start()

# --- Keyboard listener ---
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()

rec_thread.join()
trans_thread.join()
print("Exiting cleanly.")