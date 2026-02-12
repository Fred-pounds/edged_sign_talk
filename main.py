import cv2
import time
import sys
import os
import argparse

from hand_tracker import HandTracker
# from model_loader import ModelLoader
# from word_builder import WordBuilder
from speech_engine import SpeechEngine
from gesture_recognizer import GestureRecognizer

def main():
    parser = argparse.ArgumentParser(description="Sign2Speech - Edge AI Fingerspelling Translator")
    parser.add_argument("--source", type=str, default="0", help="Video source: webcam index (0) or URL (http://...)")
    args = parser.parse_args()

    print("Initializing Sign2Speech (Whole Word Mode)...")
    
    # Initialize components
    try:
        # Check if model exists
        if not os.path.exists("lstm_model.tflite"):
            print("ERROR: 'lstm_model.tflite' not found in current directory.")
            print("Please run train_lstm.py first.")
            return

        tracker = HandTracker(detection_con=0.7, max_hands=1)
        # model = ModelLoader(model_path="model.tflite")
        # wb = WordBuilder(stability_duration=1.0)
        recognizer = GestureRecognizer()
        speech = SpeechEngine()
        
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # Initialize Camera
    source = args.source
    if source.isdigit():
        source = int(source)
    
    print(f"Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    
    # Set resolution to 640x480 (Only works for local webcams usually, safely ignored for streams)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting Main Loop. Press 'q' or 'Esc' to exit.")
    
    last_word = ""
    cooldown_counter = 0

    try:
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # 1. Detect Hand
            tracker.find_hands(img, draw=True)
            lm_list = tracker.get_landmark_data()

            current_word = "Listening..."
            
            if lm_list:
                # 2. Recognize Gesture
                try:
                    prediction = recognizer.process_landmarks(lm_list)
                    
                    if prediction:
                        current_word = f"Recognized: {prediction}"
                        
                        # Simple debounce/cooldown
                        if prediction != last_word or cooldown_counter > 30:
                            print(f"Matched Word: {prediction}")
                            speech.say(prediction)
                            last_word = prediction
                            cooldown_counter = 0
                        
                    if last_word == prediction:
                        cooldown_counter += 1
                        
                except Exception as e:
                    print(f"Prediction error: {e}")
            else:
                recognizer.clear()
                current_word = "No Hand"

            # UI Display
            # Draw distinct box for stats
            cv2.rectangle(img, (0, 0), (640, 80), (0, 0, 0), cv2.FILLED)
            
            # Display Prediction
            cv2.putText(img, current_word, (10, 50), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

            cv2.imshow("Sign2Speech - Whole Word (LSTM)", img)

            key = cv2.waitKey(1)
            if key == ord('q') or key == 27: # q or ESC
                break

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        speech.cleanup()
        print("Application closed.")

if __name__ == "__main__":
    main()
