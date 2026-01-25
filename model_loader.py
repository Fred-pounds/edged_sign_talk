import numpy as np

# Try importing tflite_runtime, fallback to tensorflow if not available
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        raise ImportError("Neither tflite_runtime nor tensorflow is installed.")

class ModelLoader:
    """
    Handles loading the TFLite model and running inference.
    """
    def __init__(self, model_path="model.tflite"):
        """
        Load the TFLite model.
        
        Args:
            model_path (str): Path to the .tflite model file.
        """
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Label mapping for 26 classes (A-Z)
            self.labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
            
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            raise

    def predict(self, landmarks):
        """
        Run inference on the provided landmarks.
        
        Args:
            landmarks (list): List of 63 float values (flattend x, y, z).
            
        Returns:
            str: Predicted letter (A-Z).
        """
        input_data = np.array(landmarks, dtype=np.float32)
        
        expected_shape = self.input_details[0]['shape']
        expected_cols = expected_shape[1]
        
        # If model expects 42 features (21 * 2), convert 63 (21 * 3) to 42 by dropping Z
        if expected_cols == 42 and len(input_data) == 63:
            # Reshape to (21, 3), take first 2 cols (x,y), flatten
            input_data = input_data.reshape(21, 3)[:, :2].flatten()
        
        # Add batch dimension
        input_data = np.array([input_data], dtype=np.float32)

        # Handle any other dimension mismatch (e.g. padding if needed, though we expect 42 now)
        if input_data.shape[1] != expected_cols:
             print(f"Warning: Input shape {input_data.shape} does not match expected {expected_shape}. Padding/Truncating.")
             # Fallback padding/truncating logic could go here if needed
             if input_data.shape[1] < expected_cols:
                 padded = np.zeros((1, expected_cols), dtype=np.float32)
                 padded[:, :input_data.shape[1]] = input_data
                 input_data = padded
             else:
                 input_data = input_data[:, :expected_cols]

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        prediction_index = np.argmax(output_data)
        
        return self.labels[prediction_index]
