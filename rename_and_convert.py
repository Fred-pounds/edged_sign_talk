"""
Rename and convert .MOV files in data/raw to .mp4 with class-based naming.
e.g. data/raw/hello/IMG_1124.MOV -> data/raw/hello/hello_8.mp4
"""
import os
import cv2

DATA_PATH = os.path.join("data", "raw")

def get_next_index(action_path, action):
    """Find the next available index for this action class."""
    existing = [f for f in os.listdir(action_path) if f.startswith(action + "_") and f.endswith(".mp4")]
    if not existing:
        return 1
    indices = []
    for f in existing:
        try:
            idx = int(f.replace(action + "_", "").replace(".mp4", ""))
            indices.append(idx)
        except ValueError:
            pass
    return max(indices) + 1 if indices else 1

def convert_mov_to_mp4(mov_path, mp4_path):
    """Convert a MOV file to MP4 using OpenCV."""
    cap = cv2.VideoCapture(mov_path)
    if not cap.isOpened():
        print(f"  ERROR: Could not open {mov_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"  Converted: {os.path.basename(mov_path)} -> {os.path.basename(mp4_path)} ({frame_count} frames, {width}x{height})")
    return True

def main():
    actions = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])

    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        mov_files = sorted([f for f in os.listdir(action_path) if f.upper().endswith(".MOV")])

        if not mov_files:
            print(f"{action}: No .MOV files to convert")
            continue

        print(f"\n{action}: {len(mov_files)} .MOV files to convert")
        next_idx = get_next_index(action_path, action)

        for mov_file in mov_files:
            mov_path = os.path.join(action_path, mov_file)
            new_name = f"{action}_{next_idx}.mp4"
            mp4_path = os.path.join(action_path, new_name)

            success = convert_mov_to_mp4(mov_path, mp4_path)
            if success:
                os.remove(mov_path)
                next_idx += 1

    # Print final counts
    print("\n--- Final Dataset Summary ---")
    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        mp4_count = len([f for f in os.listdir(action_path) if f.endswith(".mp4")])
        mov_count = len([f for f in os.listdir(action_path) if f.upper().endswith(".MOV")])
        print(f"{action}: {mp4_count} mp4 files, {mov_count} MOV files remaining")

if __name__ == "__main__":
    main()
