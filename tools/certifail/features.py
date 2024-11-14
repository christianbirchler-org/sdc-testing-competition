import math
import pandas as pd


def extract_coordinates(sd_test_case):
    coordinates = []
    if hasattr(sd_test_case, 'roadPoints') and sd_test_case.roadPoints:
        for road_point in sd_test_case.roadPoints:
            coordinates.append({'x': road_point.x, 'y': road_point.y})
    return coordinates

# Function to normalize the angle difference
def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle



def calculate_features(sd_test_case):
    # Extract road points
    road_points = extract_coordinates(sd_test_case)

    # Check if there are enough points to calculate features
    if len(road_points) < 2:
        raise ValueError("Not enough road points to calculate features.")

    # Feature initialization
    road_length = 0
    direct_length = math.sqrt(
        (road_points[-1]['x'] - road_points[0]['x'])**2 + (road_points[-1]['y'] - road_points[0]['y'])**2
    )
    max_turn_angle = 0
    total_turn_angle = 0
    max_right_turn = 0
    max_left_turn = 0
    total_right_turns = 0
    total_left_turns = 0
    max_jerk = 0
    

    prev_turn_angle = None

    # Iterate through road points to calculate features
    for i in range(1, len(road_points) - 1):
        # Extract coordinates for three consecutive points
        x1, y1 = road_points[i-1]['x'], road_points[i-1]['y']
        x2, y2 = road_points[i]['x'], road_points[i]['y']
        x3, y3 = road_points[i+1]['x'], road_points[i+1]['y']

        # Calculate road segment length
        road_length += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Calculate angles for direction changes
        angle1 = math.atan2(y2 - y1, x2 - x1)
        angle2 = math.atan2(y3 - y2, x3 - x2)
        turn_angle = normalize_angle(angle2 - angle1)

        # Update total and maximum angle change
        total_turn_angle += abs(turn_angle)
        max_turn_angle = max(max_turn_angle, abs(turn_angle))

        # Determine the direction of the turn
        if turn_angle > 0:  # Left turn
            total_left_turns += 1
            max_left_turn = max(max_left_turn, turn_angle)
        elif turn_angle < 0:  # Right turn
            total_right_turns += 1
            max_right_turn = max(max_right_turn, abs(normalize_angle(turn_angle)))

        if prev_turn_angle is not None:
            angle_change = turn_angle - prev_turn_angle
            max_jerk = max(max_jerk,abs(angle_change))

        prev_turn_angle = turn_angle

    # Calculate the ratio of the actual road length to the direct length
    length_ratio = road_length / direct_length if direct_length != 0 else 0

    features_df = pd.DataFrame({
        'road_length': [road_length],
        'direct_length': [direct_length],
        'length_ratio': [length_ratio],
        'max_turn_angle': [max_turn_angle],
        'total_turn_angle': [total_turn_angle],
        'max_right_turn': [max_right_turn],
        'max_left_turn': [max_left_turn],
        'total_right_turns': [total_right_turns],
        'total_left_turns': [total_left_turns],
        'max_jerk': [max_jerk]
    })

    return features_df

 

