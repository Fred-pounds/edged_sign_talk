import os

DATA_PATH = "data/raw"

def check_balance():
    if not os.path.exists(DATA_PATH):
        print(f"{DATA_PATH} not found.")
        return

    actions = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    actions.sort()
    
    print(f"Found {len(actions)} actions.")
    
    counts = {}
    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        files = [f for f in os.listdir(action_path) if f.endswith(".mp4")]
        counts[action] = len(files)
        print(f"{action}: {len(files)} videos")
        
    values =  options = list(counts.values())
    if not values:
        print("No data found.")
        return

    min_count = min(values)
    max_count = max(values)
    avg_count = sum(values) / len(values)
    
    print("\nSummary:")
    print(f"Min videos: {min_count}")
    print(f"Max videos: {max_count}")
    print(f"Avg videos: {avg_count:.2f}")

    if max_count - min_count > 5: # Arbitrary threshold
        print("\nWarning: Data imbalance detected.")
    else:
        print("\nData seems fairly balanced.")

if __name__ == "__main__":
    check_balance()
