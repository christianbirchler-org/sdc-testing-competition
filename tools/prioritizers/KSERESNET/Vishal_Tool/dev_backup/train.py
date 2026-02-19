import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import time

# --- CONFIGURATION ---
DATA_PATH = "../../../data/sdc-test-data.json" 
MAX_POINTS = 200
INPUT_CHANNELS = 4  # Changed from 2 to 4 (Velocity + Acceleration)
BATCH_SIZE = 32     
EPOCHS = 40         

# --- 1. NOVELTY: KINEMATIC FEATURE EXTRACTION (The "Math" Part) ---
def process_road_points(points):
    coords = np.array([[p['x'], p['y']] for p in points])
    
    # Handle empty or short paths
    if len(coords) < 3: 
        return np.zeros((MAX_POINTS, INPUT_CHANNELS))
    
    # A. Calculate Velocity (First Derivative)
    deltas = coords[1:] - coords[:-1]
    
    # B. Calculate Acceleration (Second Derivative) - THIS IS THE NEW MATH
    # This explicitly captures "Curvature" and "Force"
    accel_raw = deltas[1:] - deltas[:-1]
    # Pad accel to match deltas length (add 0 at start)
    accel = np.vstack((np.zeros((1, 2)), accel_raw))

    # C. Normalize (Math scaling)
    deltas[:, 0] = (deltas[:, 0] - (-0.02)) / 0.5
    deltas[:, 1] = (deltas[:, 1] - 0.20) / 0.5
    accel = accel * 5.0  # Scale up acceleration to make it visible to the AI

    # D. Feature Fusion (Velocity + Acceleration)
    # Result shape: [Length, 4]
    features = np.hstack((deltas, accel))

    # Padding to fixed length
    if len(features) > MAX_POINTS: 
        return features[:MAX_POINTS]
    else:
        padding = np.zeros((MAX_POINTS - len(features), INPUT_CHANNELS))
        return np.vstack((features, padding))

class BalancedRoadDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f: raw_data = json.load(f)
        safe, crash = [], []
        for item in raw_data:
            # Pass dictionary list to processing function
            seq = process_road_points(item['road_points'])
            if item['meta_data']['test_info']['test_outcome'] == 'FAIL': crash.append(seq)
            else: safe.append(seq)
        
        while len(crash) < len(safe): crash.append(random.choice(crash))
        
        self.features = torch.FloatTensor(np.array(safe + crash))
        self.labels = torch.FloatTensor(np.array([0.0]*len(safe) + [1.0]*len(crash))).unsqueeze(1)
        
        indices = torch.randperm(len(self.features))
        self.features = self.features[indices]
        self.labels = self.labels[indices]

    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

# --- 2. NOVELTY: SE-ATTENTION BLOCK ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SEBlock(channels) # Attention Layer
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out) 
        out += residual
        return self.relu(out)

class ResNetPredictor(nn.Module):
    def __init__(self):
        super(ResNetPredictor, self).__init__()
        # Input is now 4 Channels (dx, dy, ax, ay)
        self.input_layer = nn.Conv1d(INPUT_CHANNELS, 32, kernel_size=5, padding=2)
        self.res_block1 = ResidualBlock(32)
        self.pool1 = nn.MaxPool1d(2) 
        self.res_block2 = ResidualBlock(32)
        self.pool2 = nn.MaxPool1d(2) 
        self.res_block3 = ResidualBlock(32) 
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 50, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.pool1(x)
        x = self.res_block2(x)
        x = self.pool2(x)
        x = self.res_block3(x)
        return self.fc(x)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    dataset = BalancedRoadDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = ResNetPredictor()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.BCEWithLogitsLoss() 
    
    print(f"--- TRAINING HYBRID MODEL (Kinematics + Attention) ---")
    best_acc = 0.0
    for epoch in range(EPOCHS): 
        correct, total = 0, 0
        model.train()
        for X, y in dataloader:
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            predicted = (torch.sigmoid(logits) > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1:02d} | Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "crash_model.pth")
            
    print(f"Best Accuracy: {best_acc:.2f}%")

    # --- 2. COMPETITION METRICS BENCHMARK ---
    print("\n--- 2. BENCHMARKING (Speed Test) ---")
    
    start_init = time.time()
    benchmark_model = ResNetPredictor()
    benchmark_model.load_state_dict(torch.load("crash_model.pth"))
    benchmark_model.eval()
    end_init = time.time()
    init_time = end_init - start_init
    
    all_test_cases = dataset.features 
    start_select = time.time()
    with torch.no_grad():
        scores = benchmark_model(all_test_cases)
        sorted_indices = torch.argsort(scores.flatten(), descending=True)
    end_select = time.time()
    select_time = end_select - start_select
print(f"Total Test Cases Processed: {len(dataset)}")

print(f"-"*30)
print(f"Initialization Time : {init_time:.5f} seconds")
print(f"Selection Time      : {select_time:.5f} seconds")
print(f"-"*30)
