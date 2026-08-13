import json
import grpc
import competition_2026_pb2
import competition_2026_pb2_grpc

def get_apfd(ranked_data):
    n = len(ranked_data)
    # The official crash key we found earlier
    crashes = [i for i, d in enumerate(ranked_data) if d.get('meta_data', {}).get('test_info', {}).get('test_outcome') == 'FAIL']
    m = len(crashes)
    if m == 0 or n == 0: return 0.0
    
    sum_ranks = sum([(r + 1) for r in crashes])
    apfd = 1 - (sum_ranks / (n * m)) + (1 / (2 * n))
    return apfd

def verify():
    print("Loading data and connecting to Vishal_Tool...")
    with open('../../../data/sdc-test-data.json', 'r') as f:
        data = json.load(f)
    
    channel = grpc.insecure_channel('localhost:50051')
    stub = competition_2026_pb2_grpc.CompetitionToolStub(channel)
    
    try:
        # Handshake
        stub.Initialize(iter([competition_2026_pb2.Empty()]))
        
        # Prepare the stream
        def request_generator():
            for item in data:
                pts = [competition_2026_pb2.RoadPoint(x=float(p['x']), y=float(p['y'])) for p in item.get('road_points', [])]
                yield competition_2026_pb2.SDCTestCase(testId=str(item.get('_id')), roadPoints=pts)

        print("Sending 956 tests for prioritization...")
        responses = stub.Prioritize(request_generator())
        
        ranked_ids = [r.testId for r in responses]
        id_map = {str(item.get('_id')): item for item in data}
        ranked_data = [id_map[rid] for rid in ranked_ids if rid in id_map]
        
        score = get_apfd(ranked_data)
        print("\n" + "="*40)
        print(f"OFFICIAL VERIFIED APFD SCORE: {score:.4f}")
        print("="*40)
        
    except Exception as e:
        print(f"Connection Error: {e}\nMake sure 'python3 main.py' is running!")

if __name__ == "__main__":
    verify()
