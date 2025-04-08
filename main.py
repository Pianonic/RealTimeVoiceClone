import os
import time
import shutil
import threading
import subprocess
import torch
import torchaudio
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path

# Configuration
WATCH_FOLDER = "input_audio"
TRAINING_DATA_FOLDER = "training_data"
MODEL_SNAPSHOTS_FOLDER = "model_snapshots"
AUDIO_TO_CONVERT_FOLDER = "audio_to_convert"
OUTPUT_FOLDER = "converted_audio"
SAMPLE_RATE = 16000

# Create necessary folders
for folder in [WATCH_FOLDER, TRAINING_DATA_FOLDER, MODEL_SNAPSHOTS_FOLDER, 
               AUDIO_TO_CONVERT_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Simple voice conversion model (for demonstration)
class SimpleVoiceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, kernel_size=15, stride=1, padding=7),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, kernel_size=15, stride=1, padding=7),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv1d(64, 32, kernel_size=15, stride=1, padding=7),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 1, kernel_size=15, stride=1, padding=7),
            torch.nn.Tanh(),
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def convert_voice(self, audio_tensor):
        # Simple voice conversion by applying the model
        with torch.no_grad():
            self.eval()
            return self(audio_tensor)

# Audio preprocessing functions
def preprocess_audio(file_path):
    """Process audio file for training or conversion"""
    audio, sr = torchaudio.load(file_path)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    # Resample if needed
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        audio = resampler(audio)
    
    # Normalize audio
    audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
    
    return audio

def train_model(model, training_files, epochs=10):
    """Train the model on the given audio files"""
    print(f"Training model on {len(training_files)} files...")
    
    # Simple training loop for demonstration
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for file_path in training_files:
            # Load and preprocess audio
            audio = preprocess_audio(file_path)
            
            # Simple reconstruction training (you'd use different targets in real voice conversion)
            optimizer.zero_grad()
            output = model(audio)
            loss = loss_fn(output, audio)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(training_files):.6f}")
    
    print("Training complete!")
    return model

def save_model_snapshot(model, snapshot_name):
    """Save a snapshot of the current model"""
    save_path = os.path.join(MODEL_SNAPSHOTS_FOLDER, f"{snapshot_name}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model snapshot saved to {save_path}")
    return save_path

def load_latest_model():
    """Load the latest model snapshot"""
    snapshots = list(Path(MODEL_SNAPSHOTS_FOLDER).glob("*.pt"))
    if not snapshots:
        print("No model snapshots found. Using new model.")
        return SimpleVoiceModel()
    
    latest_snapshot = max(snapshots, key=os.path.getctime)
    print(f"Loading model from {latest_snapshot}")
    
    model = SimpleVoiceModel()
    model.load_state_dict(torch.load(str(latest_snapshot)))
    return model

def convert_audio_file(model, input_file, output_file):
    """Convert an audio file using the model"""
    print(f"Converting {input_file} to {output_file}")
    
    # Load and preprocess input audio
    audio = preprocess_audio(input_file)
    
    # Apply the model for conversion
    with torch.no_grad():
        converted_audio = model.convert_voice(audio)
    
    # Save the converted audio
    torchaudio.save(output_file, converted_audio, SAMPLE_RATE)
    print(f"Conversion complete: {output_file}")

# File system event handler
class AudioFileHandler(FileSystemEventHandler):
    def __init__(self):
        self.model = load_latest_model()
        self.training_in_progress = False
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        if not event.src_path.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
            return
        
        # Get clean file path
        file_path = event.src_path
        file_name = os.path.basename(file_path)
        
        print(f"New audio file detected: {file_name}")
        
        # Wait a moment to ensure file is completely written
        time.sleep(1)
        
        if os.path.dirname(file_path) == os.path.abspath(WATCH_FOLDER):
            print("as")
            # Copy to training data
            dest_path = os.path.join(TRAINING_DATA_FOLDER, file_name)
            shutil.copy2(file_path, dest_path)
            print(f"Copied {file_name} to training data folder")
            
            # Start training in a separate thread
            if not self.training_in_progress:
                training_thread = threading.Thread(target=self.train_and_update_model)
                training_thread.start()
        
        elif os.path.dirname(file_path) == os.path.abspath(AUDIO_TO_CONVERT_FOLDER):
            # Convert this file using current model
            output_path = os.path.join(OUTPUT_FOLDER, f"converted_{file_name}")
            convert_audio_file(self.model, file_path, output_path)
    
    def train_and_update_model(self):
        """Train the model with all available training data"""
        self.training_in_progress = True
        
        try:
            # Get all training files
            training_files = [os.path.join(TRAINING_DATA_FOLDER, f) 
                             for f in os.listdir(TRAINING_DATA_FOLDER)
                             if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]
            
            if not training_files:
                print("No training files available.")
                self.training_in_progress = False
                return
            
            # Train the model
            self.model = train_model(self.model, training_files)
            
            # Save a snapshot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_model_snapshot(self.model, f"voice_model_{timestamp}")
            
            # Convert any existing files in the conversion folder
            for file_name in os.listdir(AUDIO_TO_CONVERT_FOLDER):
                if file_name.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                    input_path = os.path.join(AUDIO_TO_CONVERT_FOLDER, file_name)
                    output_path = os.path.join(OUTPUT_FOLDER, f"converted_{file_name}")
                    convert_audio_file(self.model, input_path, output_path)
            
        except Exception as e:
            print(f"Error during training: {e}")
        
        self.training_in_progress = False

def main():
    print("Starting Voice Cloning Application")
    print(f"Watching for new audio files in: {os.path.abspath(WATCH_FOLDER)}")
    print(f"Place files to convert in: {os.path.abspath(AUDIO_TO_CONVERT_FOLDER)}")
    print(f"Converted files will appear in: {os.path.abspath(OUTPUT_FOLDER)}")
    
    # Set up the file system observer
    event_handler = AudioFileHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_FOLDER, recursive=False)
    observer.schedule(event_handler, AUDIO_TO_CONVERT_FOLDER, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()