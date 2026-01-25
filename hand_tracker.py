import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    """
    Wrapper for MediaPipe Hand Landmarker (Tasks API) to detect a single hand and extract landmarks.
    """
    def __init__(self, mode=False, max_hands=1, detection_con=0.5, track_con=0.5, model_path="hand_landmarker.task"):
        """
        Initialize the MediaPipe Hand Landmarker.
        """
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_con,
            min_hand_presence_confidence=track_con
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None
        
        # Define connections for manual drawing (standard MediaPipe hand connections)
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),           # Index
            (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
            (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
            (0, 17)                                   # Wrist to Pinky
        ]

    def find_hands(self, img, draw=True):
        """
        Process the image to find hands.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        self.results = self.detector.detect(mp_image)

        if self.results.hand_landmarks:
            for hand_lms in self.results.hand_landmarks:
                if draw:
                    self._draw_landmarks_manual(img, hand_lms)
        return img

    def _draw_landmarks_manual(self, img, landmarks):
        """
        Manually draw landmarks and connections since mp.solutions.drawing_utils is unavailable.
        """
        h, w, c = img.shape
        # Convert normalized coordinates to pixel coordinates
        points = []
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            points.append((cx, cy))

        # Draw connections
        for p1_idx, p2_idx in self.connections:
            if p1_idx < len(points) and p2_idx < len(points):
                cv2.line(img, points[p1_idx], points[p2_idx], (255, 255, 255), 3)

        # Draw points
        for cx, cy in points:
             cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

    def get_landmark_data(self):
        """
        Extract normalized (x, y, z) coordinates for the first detected hand.
        """
        if self.results and self.results.hand_landmarks:
            # Only process the first hand
            my_hand = self.results.hand_landmarks[0]
            lm_list = []
            for lm in my_hand:
                # Normalized coordinates
                lm_list.extend([lm.x, lm.y, lm.z])
            return lm_list
        return None

    def close(self):
        """Release MediaPipe resources."""
        self.detector.close()
