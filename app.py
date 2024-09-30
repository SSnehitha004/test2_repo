from flask import Flask, request, render_template, send_file
import os
import time
import torch
from pathlib import Path
from TTS.utils.synthesizer import Synthesizer
import fitz  # PyMuPDF
import wave

app = Flask(__name__)

# Define paths
base_model_path = Path(r"C:\Users\HP\Desktop\nlp\trump.pth")
juice_wrld_path = Path(r"C:\Users\HP\Desktop\nlp\juice-wrld.pth")
config_path = Path(r"C:\Users\HP\Desktop\nlp\models_config.json")
output_path = Path(r"C:\Users\HP\Desktop\nlp\results")
os.makedirs(output_path, exist_ok=True)

# Set CUDA usage based on availability
use_cuda = torch.cuda.is_available()

# Function to synthesize text to audio and save the output
def synthesize(text: str, voice: str, speed: float):
    start_time = time.time()  # Start time for synthesis
    if voice == 'juice_wrld':
        model_file = juice_wrld_path
    else:
        model_file = base_model_path

    if not model_file.exists():
        raise FileNotFoundError(f"Model file {model_file} does not exist.")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} does not exist.")

    print(f"Starting text-to-speech synthesis with {voice} voice and speed {speed}...")
    synthesizer = Synthesizer(tts_config_path=config_path, tts_checkpoint=model_file, use_cuda=use_cuda)
    wav = synthesizer.tts(text, speed=speed)
    output_filename = f"{int(time.time())}_{voice}.wav"
    output_file_path = output_path / output_filename
    synthesizer.save_wav(wav, output_file_path)
    
    end_time = time.time()  # End time for synthesis
    print(f"Synthesis completed in {end_time - start_time} seconds.")
    return output_file_path

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    start_time = time.time()  # Start time for PDF extraction
    text = ""
    print(f"Starting text extraction from PDF: {pdf_path}")
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    end_time = time.time()  # End time for PDF extraction
    print(f"Text extraction completed in {end_time - start_time} seconds.")
    return text

# Function to get audio duration
def get_audio_duration(audio_file):
    start_time = time.time()  # Start time for duration calculation
    with wave.open(str(audio_file), 'r') as audio:
        frames = audio.getnframes()
        rate = audio.getframerate()
        duration = frames / float(rate)
    end_time = time.time()  # End time for duration calculation
    print(f"Audio duration calculation completed in {end_time - start_time} seconds.")
    return duration

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    start_time = time.time()  # Start time for overall process
    if 'pdf' not in request.files:
        return {"error": "No PDF file uploaded"}, 400

    pdf_file = request.files['pdf']
    pdf_path = os.path.join(output_path, pdf_file.filename)
    pdf_file.save(pdf_path)

    # Extract and synthesize
    text = extract_text_from_pdf(pdf_path)
    
    # Get selected voice and speed from the form
    selected_voice = request.form.get('voice')
    speed = float(request.form.get('speed', 1))  # Default speed is 1x
    
    audio_file_path = synthesize(text, selected_voice, speed)

    # Calculate word timings
    words = text.split()
    audio_duration = get_audio_duration(audio_file_path)
    word_timing = audio_duration / len(words)  # Approximate time per word

    end_time = time.time()  # End time for overall process
    print(f"Total processing time: {end_time - start_time} seconds.")
    
    return render_template('result.html', words=words, word_timing=word_timing, audio_file=audio_file_path.name)

@app.route('/playback/<audio_file>')
def playback(audio_file):
    return send_file(os.path.join(output_path, audio_file), as_attachment=False)

if __name__ == "__main__":
    app.run(debug=True)
