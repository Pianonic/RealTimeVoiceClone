# Continuous Voice Cloning Application

A lightweight application that automatically trains a voice conversion model from new audio samples and converts audio files using the trained model.

> **⚠️ Important Note:** This project is currently under development and may not function as described directly from the main branch.

## Features

- **Automatic Training**: Monitors a folder for new voice samples and automatically incorporates them into training
- **Incremental Learning**: Builds on previous training rather than starting from scratch
- **Easy Conversion**: Simply place audio files in a designated folder to convert them with the latest voice model
- **Model Snapshots**: Automatically saves model versions after each training session

## Installation

### Prerequisites

- Python 3.8+
- PyTorch and TorchAudio
- Watchdog library

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/voice-cloning-app.git
cd voice-cloning-app
```

2. Install dependencies:
```bash
pip install torch torchaudio watchdog numpy
```

3. Create the required directories (the app will do this automatically on first run):
```
input_audio/        # Place new voice samples here
training_data/      # Training data storage (managed by the app)
model_snapshots/    # Where model versions are saved
audio_to_convert/   # Place audio files to be converted here
converted_audio/    # Output folder for converted audio
```

## Usage

1. **Start the application**:
```bash
python voice_cloning_app.py
```

2. **Add training samples**:
   - Place voice samples of the target voice in the `input_audio` folder
   - Supported formats: WAV, MP3, OGG, FLAC
   - The application will automatically detect new files and use them for training

3. **Convert audio**:
   - Place audio files you want to convert in the `audio_to_convert` folder
   - The application will convert them using the latest trained model
   - Find the converted files in the `converted_audio` folder with the prefix "converted_"

## Advanced Configuration

Edit the following constants in `voice_cloning_app.py` to customize the application:

```python
WATCH_FOLDER = "input_audio"               # Folder to watch for new training samples
TRAINING_DATA_FOLDER = "training_data"     # Where training data is stored
MODEL_SNAPSHOTS_FOLDER = "model_snapshots" # Where model versions are saved
AUDIO_TO_CONVERT_FOLDER = "audio_to_convert" # Where to look for files to convert
OUTPUT_FOLDER = "converted_audio"          # Where converted files are saved
SAMPLE_RATE = 16000                        # Audio sample rate for processing
```

## Improving Voice Quality

The default implementation uses a simple voice conversion model for demonstration purposes. For better quality:

1. Replace the `SimpleVoiceModel` class with a more sophisticated model like So-VITS-SVC or YourTTS
2. Enhance the audio preprocessing functions to better clean and normalize input audio
3. Modify the training parameters for better voice adaptation

## Troubleshooting

- **Audio quality issues**: Try increasing the number of epochs in the `train_model` function
- **Out of memory errors**: Reduce batch sizes or use audio chunking for long files
- **Training taking too long**: Enable GPU acceleration by ensuring PyTorch is installed with CUDA support

## Future Improvements

- Discord bot integration to collect voice samples directly from voice chats
- Real-time voice conversion capabilities
- Web interface for managing the application
- Support for multiple voice pro
