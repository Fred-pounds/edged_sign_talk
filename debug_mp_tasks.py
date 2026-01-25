try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    print("Tasks API import successful")
    print(f"Vision module: {vision}")
    print(f"HandLandmarker: {vision.HandLandmarker}")
except ImportError as e:
    print(f"ImportError: {e}")
except AttributeError as e:
    print(f"AttributeError: {e}")
