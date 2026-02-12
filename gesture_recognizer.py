import numpy as np
import time

# Try importing tflite_runtime, fallback to tensorflow if not available
try:
    import tflite_runtime.interpreter as tflite
    # If using Select TF Ops, we might need to load the delegate or ensure library is loaded
    # Usually standard tflite_runtime might not support Select TF Ops.
    # If standard tflite_runtime fails, we must use tensorflow.lite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        raise ImportError("Neither tflite_runtime nor tensorflow is installed.")

class GestureRecognizer:
    """
    Handles real-time gesture recognition using an LSTM TFLite model.
    """
    def __init__(self, model_path="lstm_model.tflite", label_path="labels.txt", threshold=0.8):
        """
        Initialize the recognizer.
        """
        self.threshold = threshold
        self.sequence_length = 30
        self.sequence_buffer = [] # Buffer to store landmarks
        
        # Load Labels
        self.labels = []
        try:
            with open(label_path, "r") as f:
                self.labels = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"Warning: {label_path} not found. Using numeric output.")
            
        # Load Model
        try:
            # If using Select TF ops, we might need extra arguments for Interpreter in some versions
            # But usually it's automatic if tensorflow is installed
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"Gesture Recognizer initialized. Labels: {self.labels}")
            
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            raise

    def process_landmarks(self, landmarks):
        """
        Add landmarks to buffer and run inference if buffer is full.
        
        Args:
            landmarks (list): 63 landmarks (21 * 3)
            
        Returns:
            str: Predicted action or None if uncertainty/buffer filling
        """
        # Normalize landmarks relative to wrist (first 3 values are x,y,z of wrist)
        # We need numpy for this
        lm_np = np.array(landmarks)
        wrist = lm_np[:3]
        relative_lm = lm_np - np.tile(wrist, 21) # Subtract wrist from all
        
        self.sequence_buffer.append(relative_lm)
        
        # Maintain buffer size
        if len(self.sequence_buffer) > self.sequence_length:
            self.sequence_buffer.pop(0)
            
        # Predict if we have enough frames
        if len(self.sequence_buffer) == self.sequence_length:
            return self._predict()
            
        return None

    def _predict(self):
        """
        Run inference on the current buffer.
        """
        # Prepare input
        input_data = np.array([self.sequence_buffer], dtype=np.float32)
        
        # Check input shape
        # expected (1, 30, 63)
        
        try:
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            prediction = np.squeeze(output_data)
            
            max_index = np.argmax(prediction)
            confidence = prediction[max_index]
            
            if confidence > self.threshold:
                if self.labels:
                    return self.labels[max_index]
                return str(max_index)
            
        except Exception as e:
            print(f"Inference error: {e}")
            # If we are using Select TF ops and standard tflite_runtime, this might fail
            
        return None

    def clear(self):
        self.sequence_buffer = []
