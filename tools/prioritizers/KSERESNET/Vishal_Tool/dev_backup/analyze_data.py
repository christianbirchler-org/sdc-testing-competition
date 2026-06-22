import json
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "../../../data/sdc-test-data.json"

def analyze():
    print(f"Reading {DATA_PATH}...")
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    
    all_dx = []
    all_dy = []
    lengths = []
    
    for item in data:
        points = item['road_points']
        coords = np.array([[p['x'], p['y']] for p in points])
        
        # Calculate Deltas
        if len(coords) > 1:
            deltas = coords[1:] - coords[:-1]
            all_dx.extend(deltas[:, 0])
            all_dy.extend(deltas[:, 1])
            lengths.append(len(coords))

    all_dx = np.array(all_dx)
    all_dy = np.array(all_dy)
    
    print("-" * 30)
    print(f"ANALYSIS REPORT:")
    print(f"Total Points: {len(all_dx)}")
    print(f"Avg Road Length: {np.mean(lengths):.1f} points")
    print("-" * 30)
    print(f"DX (Horizontal Move) -> Min: {all_dx.min():.4f}, Max: {all_dx.max():.4f}, Mean: {all_dx.mean():.4f}")
    print(f"DY (Forward Move)    -> Min: {all_dy.min():.4f}, Max: {all_dy.max():.4f}, Mean: {all_dy.mean():.4f}")
    print("-" * 30)
    
    # Suggest Normalization Factor
    max_val = max(abs(all_dx.max()), abs(all_dy.max()), abs(all_dx.min()), abs(all_dy.min()))
    print(f"RECOMMENDATION: Divide inputs by {max_val:.2f} (or round up to {np.ceil(max_val)})")

if __name__ == "__main__":
    analyze()