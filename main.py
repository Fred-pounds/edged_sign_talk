import cv2
import time
import sys
import os

from hand_tracker import HandTracker
from model_loader import ModelLoader
from word_builder import WordBuilder
from speech_engine import SpeechEngine

def main():
    print("Initializing Sign2Speech...")
    
    # Initialize components
    try:
        # Check if model exists
        if not os.path.exists("model.tflite"):
            print("ERROR: 'model.tflite' not found in current directory.")
            print("Please place the Ishara TFLite model file in this folder.")
            return

        tracker = HandTracker(detection_con=0.7, max_hands=1)
        model = ModelLoader(model_path="model.tflite")
        wb = WordBuilder(stability_duration=1.0)
        speech = SpeechEngine()
        
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # Initialize Camera
    cap = cv2.VideoCapture(0)
    # Set resolution to 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting Main Loop. Press 'q' or 'Esc' to exit.")

    try:
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # 1. Detect Hand
            tracker.find_hands(img, draw=True)
            lm_list = tracker.get_landmark_data()

            current_char = ""
            
            if lm_list:
                # 2. Predict
                try:
                    predicted_char = model.predict(lm_list)
                    
                    # 3. Stabilize & Build Word
                    confirmed_char = wb.process_letter(predicted_char)
                    
                    current_char = predicted_char
                    
                    # 4. Check Vocabulary
                    matched_word = wb.check_word()
                    if matched_word:
                        print(f"Matched Word: {matched_word}")
                        speech.say(matched_word)
                        
                except Exception as e:
                    print(f"Prediction error: {e}")

            # UI Display
            # Draw distinct box for stats
            cv2.rectangle(img, (0, 0), (640, 80), (0, 0, 0), cv2.FILLED)
            
            # Display Prediction
            cv2.putText(img, f"Char: {current_char}", (10, 50), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
            
            # Display Buffer
            word_buffer = wb.get_current_word()
            cv2.putText(img, f"Word: {word_buffer}", (250, 50), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

            # Show progress bar for stability (Visual feedback)
            if lm_list and wb.last_letter:
                 # Calculate progress
                 elapsed = time.time() - wb.stable_start_time
                 progress = min(elapsed / wb.stability_duration, 1.0)
                 bar_width = int(200 * progress)
                 if bar_width > 0:
                     cv2.rectangle(img, (10, 60), (10 + bar_width, 70), (0, 255, 255), cv2.FILLED)

            cv2.imshow("Sign2Speech - Standard/Ishara", img)

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
