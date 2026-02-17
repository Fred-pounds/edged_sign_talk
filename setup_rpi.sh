#!/bin/bash

# Sign2Speech - Raspberry Pi Setup Script

echo "--- Initializing Sign2Speech Setup for Raspberry Pi ---"

# 1. Update system
echo "Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# 2. Install System Dependencies
echo "Installing system dependencies (OpenCV, MediaPipe, Speech)..."
# libgl1-mesa-glx: needed for OpenCV
# espeak-ng: needed for pyttsx3 speech engine
# libatlas-base-dev: needed for numpy/opencv
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libatlas-base-dev \
    espeak-ng \
    portaudio19-dev

# 3. Setup Virtual Environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# 4. Install Python Dependencies
echo "Installing Python packages..."
pip install --upgrade pip
pip install opencv-python mediapipe numpy pyttsx3 requests

# 5. Handle TFLite Runtime
# On RPI, it's often better to use tflite-runtime instead of the full tensorflow package
echo "Attempting to install tflite-runtime..."
pip install tflite-runtime || echo "tflite-runtime install failed, will fallback to tensorflow if needed."

# Check if tensorflow is in requirements but not installed
if ! python3 -c "import tensorflow" &> /dev/null && ! python3 -c "import tflite_runtime" &> /dev/null; then
    echo "Neither tensorflow nor tflite-runtime found. Installing tensorflow-cpu (might take a while)..."
    pip install tensorflow-cpu
fi

echo "--- Setup Complete ---"
echo "To run the application:"
echo "source .venv/bin/activate"
echo "python main.py --headless --source http://<ESP32_IP>:81/stream"
