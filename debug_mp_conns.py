from mediapipe.tasks.python import vision
try:
    print(f"HAND_CONNECTIONS: {vision.HandLandmarker.HAND_CONNECTIONS}")
except AttributeError:
    print("HAND_CONNECTIONS not found in HandLandmarker")

try:
    # Try finding it in the module
    print(f"Module connections: {vision.HandLandmarkerConnections}")
except AttributeError:
    print("HandLandmarkerConnections not found")
