import mediapipe as mp
print(f"Mediapipe file: {mp.__file__}")
try:
    print(f"Available attributes: {dir(mp)}")
    print(f"Solutions: {mp.solutions}")
except AttributeError as e:
    print(f"Error: {e}")
