# Sign2Speech - Edge AI Fingerspelling Translator

This project converts ASL fingerspelling into spoken words using a TensorFlow Lite model. It is designed to run offline on edge devices like the Raspberry Pi 4.

## Features
- **Offline Inference**: Uses TFLite implementation of the Ishara model.
- **Hand Tracking**: MediaPipe Hands for 21-point skeletal extraction.
- **Stability Filter**: Prevents jitter by requiring letters to be held for ~1 second.
- **Vocabulary Matching**: Automatically speaks recognized words (HELLO, HI, YES, NO, HELP).
- **Text-to-Speech**: Integrated offline TTS engine.

## Requirements
- Python 3.7+
- Webcam

## Installation

## Installation

### 1. Install Dependencies

**On Laptop (Windows/Mac/Linux Dev)**:
If `tflite-runtime` fails to install via pip, install the full TensorFlow package instead. The code automatically detects which one is available.
```bash
# Option A: Try default requirements
pip install -r requirements.txt

# Option B: If Option A fails on 'tflite-runtime', run:
pip install opencv-python mediapipe numpy pyttsx3 tensorflow
```

**On Raspberry Pi (Edge Deployment)**:
```bash
# Update system and install TTS engine (espeak)
sudo apt-get update && sudo apt-get install -y espeak libespeak1

# Install Python requirements
pip install -r requirements.txt
```

### 2. Add the Model
**Crucial Step**: Place your trained Ishara `model.tflite` file in this directory (`sign2speech/`). 
The model must accept input shape `(1, 63)` (flattened landmarks) or `(1, 21, 3)` and output 26 classes.

## Usage

Run the main application:
```bash
python main.py
```

- **Quit**: Press 'q' or 'Esc' to exit.
- **Operate**: Hold your hand up to the camera. Spell words letter by letter. Hold each letter until the progress bar fills to confirm it.
