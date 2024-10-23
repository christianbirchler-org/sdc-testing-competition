import joblib
import pandas as pd

# Load the saved model from the file
rf_model_local = joblib.load('best_random_forest_model.pkl')


# features = {'road_length': 207.68216352714214, 'direct_length': 50.19545128406332, 'length_ratio': 4.137469794859274, 'max_turn_angle': 0.061692289330044336, 'total_turn_angle': 6.06757852774544, 'max_right_turn': 0.061692289330044336, 'max_left_turn': 0.029905353722017747, 'total_right_turns': 141, 'total_left_turns': 54, 'max_jerk': 0.006781182773458516}

def predict_outcome(features):
    
    X_new = features

    # Make predictions with the loaded model
    y_pred_local = rf_model_local.predict(X_new)

    return int(y_pred_local[0])


# Load the saved model from the file






# Load the saved model from the file