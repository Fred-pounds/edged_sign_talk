import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

PROCESSED_PATH = os.path.join("data", "processed")
SEQUENCE_LENGTH = 30 # Fixed length for input sequences
MODEL_PATH = "lstm_model.tflite"

def load_data():
    sequences = []
    labels = []
    actions = []
    
    # Get list of actions
    try:
        actions = sorted([d for d in os.listdir(PROCESSED_PATH) if os.path.isdir(os.path.join(PROCESSED_PATH, d))])
    except FileNotFoundError:
        print("Processed data directory not found. Run process_dataset.py first.")
        return None, None, None

    label_map = {label: num for num, label in enumerate(actions)}
    
    print(f"Loading data for actions: {actions}")
    
    for action in actions:
        action_path = os.path.join(PROCESSED_PATH, action)
        for file_name in os.listdir(action_path):
            if file_name.endswith(".npy"):
                file_path = os.path.join(action_path, file_name)
                sequence = np.load(file_path)
                
                # Pad or Truncate to SEQUENCE_LENGTH
                if len(sequence) < SEQUENCE_LENGTH:
                    # Pad with zeros (or repeat last frame)
                    padding = np.zeros((SEQUENCE_LENGTH - len(sequence), 63))
                    sequence = np.concatenate([sequence, padding])
                elif len(sequence) > SEQUENCE_LENGTH:
                    # Truncate (take last N frames often better for end of sign)
                    # For now, let's take uniform sapling or just first N
                    # Taking first N is risky if video has lead-in.
                    # Let's take the middle slice
                    start = (len(sequence) - SEQUENCE_LENGTH) // 2
                    sequence = sequence[start:start+SEQUENCE_LENGTH]
                
                sequences.append(sequence)
                labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    return X, y, actions

def train_model():
    X, y, actions = load_data()
    if X is None: return
    
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    model = Sequential()
    # unroll=True is critical for TFLite compatibility without Flex delegate!
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, 63), unroll=True))
    model.add(LSTM(128, return_sequences=False, activation='relu', unroll=True))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(actions), activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    # Train
    model.fit(X, y, epochs=200, callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
    
    model.summary()
    
    # Save as TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Enable optimizations for size
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # With unroll=True, we should be able to convert using standard TFLite ops
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    
    tflite_model = converter.convert()
    
    with open(MODEL_PATH, "wb") as f:
        f.write(tflite_model)
    
    print(f"Model saved to {MODEL_PATH}")
    
    # Save labels for inference
    with open("labels.txt", "w") as f:
        for action in actions:
            f.write(action + "\n")

if __name__ == "__main__":
    train_model()
