import grpc

from concurrent import futures

import time

import numpy as np

import torch

import torch.nn as nn

import competition_2026_pb2 as pb2

import competition_2026_pb2_grpc as pb2_grpc

import gc



# --- CONFIGURATION ---

MAX_POINTS = 200

INPUT_CHANNELS = 4 



# --- YOUR KINEMATIC MATH ---

def process_road_points(points):

    coords = np.array([[float(p.x), float(p.y)] for p in points])

    if len(coords) < 3: 

        return np.zeros((MAX_POINTS, INPUT_CHANNELS))

    

    # Velocity

    deltas = coords[1:] - coords[:-1]

    # Acceleration

    accel_raw = deltas[1:] - deltas[:-1]

    accel = np.vstack((np.zeros((1, 2)), accel_raw))



    # YOUR SECRET NORMALIZATION

    deltas[:, 0] = (deltas[:, 0] - (-0.02)) / 0.5

    deltas[:, 1] = (deltas[:, 1] - 0.20) / 0.5

    accel = accel * 5.0 



    features = np.hstack((deltas, accel))

    if len(features) > MAX_POINTS: 

        return features[:MAX_POINTS]

    else:

        padding = np.zeros((MAX_POINTS - len(features), INPUT_CHANNELS))

        return np.vstack((features, padding))



# --- ARCHITECTURE COMPONENTS ---

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

        self.se = SEBlock(channels)

    def forward(self, x):

        res = x 

        out = self.relu(self.bn1(self.conv1(x)))

        out = self.se(self.bn2(self.conv2(out)))

        return self.relu(out + res)



class ResNetPredictor(nn.Module):

    def __init__(self):

        super(ResNetPredictor, self).__init__()

        self.input_layer = nn.Conv1d(INPUT_CHANNELS, 32, kernel_size=5, padding=2)

        self.res_block1 = ResidualBlock(32)

        self.pool1 = nn.MaxPool1d(2) 

        self.res_block2 = ResidualBlock(32)

        self.pool2 = nn.MaxPool1d(2) 

        self.res_block3 = ResidualBlock(32) 

        

        # --- THE MODIFIED SEQUENCE ---

        self.fc = nn.Sequential(

            nn.Flatten(),           

            nn.Linear(32 * 50, 64), 

            nn.ReLU(),              

            nn.Dropout(0.3),        

            nn.Linear(64, 1)        

        )



    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.pool1(self.res_block1(self.input_layer(x)))

        x = self.pool2(self.res_block2(x))

        x = self.res_block3(x)

        return self.fc(x)



# --- SERVICE ---

class MyPrioritizer(pb2_grpc.CompetitionToolServicer):

    def __init__(self):

        print(" Loading Hybrid Kinematic SE-ResNet...")

        self.model = ResNetPredictor() 

        try:

            self.model.load_state_dict(torch.load("crash_model.pth", map_location='cpu'))

            self.model.eval()

            print("Brain Loaded Successfully! Index sequence is perfectly matched.")

        except Exception as e:

            print(f"ERROR: {e}")



    def Name(self, request, context):

        return pb2.NameReply(name="Vishal-ResNet-Hybrid")



    def Initialize(self, request_iterator, context):

        for _ in request_iterator: pass

        return pb2.InitializationReply(ok=True)



    def Prioritize(self, request_iterator, context):

        scored_tests = []

        with torch.no_grad():

            for test in request_iterator:

                features = process_road_points(test.roadPoints)

                f_tensor = torch.FloatTensor(features).unsqueeze(0)

                score = self.model(f_tensor).item()

                scored_tests.append((str(test.testId), score))

        

        scored_tests.sort(key=lambda x: x[1], reverse=True)

        for tid, _ in scored_tests:

            yield pb2.PrioritizationReply(testId=tid)

        gc.collect()



def serve():

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))

    pb2_grpc.add_CompetitionToolServicer_to_server(MyPrioritizer(), server)

    server.add_insecure_port('[::]:50051')

    server.start()

    print("📡 Tool listening on 50051...")

    server.wait_for_termination()



if __name__ == '__main__':

    serve()
