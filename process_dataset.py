import os
import cv2
import numpy as np
import mediapipe as mp
# We will reuse the HandTracker class but need to make sure we can import it correctly
# assuming this script is in the root directory
from hand_tracker import HandTracker

DATA_PATH = os.path.join("data", "raw")
PROCESSED_PATH = os.path.join("data", "processed")
SEQUENCE_LENGTH = 30 # Number of frames to use for training (will truncate/pad)

def create_folders(actions):
    for action in actions:
        try:
            os.makedirs(os.path.join(PROCESSED_PATH, action), exist_ok=True)
        except OSError as e:
            print(f"Error creating directory for {action}: {e}")

def extract_landmarks(tracker, image):
    # Use existing tracker method but modify to return normalized relative coordinates
    # HandTracker.get_landmark_data returns absolute normalized (0-1) coordinates
    # We want them relative to the wrist (point 0) for better translation invariance
    tracker.find_hands(image, draw=False)
    results = tracker.results
    
    if results and results.hand_landmarks:
        hand_landmarks = results.hand_landmarks[0] # Assume 1 hand
        
        # Convert to numpy array
        landmarks = []
        for lm in hand_landmarks:
            landmarks.append([lm.x, lm.y, lm.z])
        landmarks = np.array(landmarks)
        
        # Normalize relative to wrist (landmark 0)
        wrist = landmarks[0]
        relative_landmarks = landmarks - wrist
        
        # Flatten
        return relative_landmarks.flatten()
    
    return np.zeros(21*3) # Return zeroes if no hand detected

def process_videos():
    actions = [name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))]
    create_folders(actions)
    
    tracker = HandTracker(max_hands=1)
    
    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        video_files = [f for f in os.listdir(action_path) if f.endswith(".mp4")]
        
        print(f"Processing action: {action} ({len(video_files)} videos)")
        
        for video_file in video_files:
            video_path = os.path.join(action_path, video_file)
            cap = cv2.VideoCapture(video_path)
            
            frames = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                landmarks = extract_landmarks(tracker, frame)
                frames.append(landmarks)
            
            cap.release()
            
            if not frames:
                print(f"  Warning: No frames extracted from {video_file}")
                continue

            # Save sequence
            # We save the raw sequence length here; padding/truncating happens in training
            npy_path = os.path.join(PROCESSED_PATH, action, video_file.replace(".mp4", ""))
            np.save(npy_path, np.array(frames))
            print(f"  Saved {npy_path}.npy (Frames: {len(frames)})")

if __name__ == "__main__":
    process_videos()
