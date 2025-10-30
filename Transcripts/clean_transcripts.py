import re
import pathlib
from tkinter import Tk, filedialog

# Folder containing the VTT files
folder = pathlib.Path("Transcripts")

# Hide main Tk window
Tk().withdraw()

# File picker opens in the Transcripts folder
files = filedialog.askopenfilenames(
    title="Select VTT files to clean",
    initialdir=folder,
    filetypes=[("VTT files", "*.vtt")]
)

for f in files:
    vtt_path = pathlib.Path(f)
    text = vtt_path.read_text(encoding="utf-8")

    # Clean transcript
    clean = re.sub(r"WEBVTT.*?\n", "", text, flags=re.S)
    clean = re.sub(r"\d+\n\d{2}:\d{2}:\d{2}\.\d{3} --> .*?\n", "", clean)
    clean = re.sub(r"\n{2,}", "\n", clean).strip()

    # Save as .txt in same folder
    output_path = vtt_path.with_suffix(".txt")
    output_path.write_text(clean, encoding="utf-8")
    print(f"Saved cleaned transcript: {output_path.name}")

print("Done.")