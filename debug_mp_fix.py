import mediapipe as mp
try:
    import mediapipe.python.solutions as solutions
    print("Direct import of solutions successful")
    print(f"Solutions hands: {solutions.hands}")
except ImportError as e:
    print(f"ImportError: {e}")
except AttributeError as e:
    print(f"AttributeError: {e}")
