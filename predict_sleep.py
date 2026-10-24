"""
Standalone Production Inference Module for Sleep Quality Prediction
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

def predict_sleep_quality(user_inputs: dict, models_dir='models/sleep'):
    """
    Predicts Sleep Quality Score (1.0 to 10.0) from raw user inputs.
    
    Expected keys:
      - sleep_duration_hrs (float, e.g. 7.5)
      - wake_episodes_per_night (int/float, e.g. 2)
      - stress_score (float, 0-10, e.g. 4.0)
      - is_healthy_mental (0 or 1, or string 'Healthy'/'Anxiety'/'Depression'/'Both')
      - has_sleep_disorder (0 or 1, or string 'Healthy'/'Mild'/'Moderate'/'Severe')
      - shift_work (0 or 1)
      - sleep_latency_mins (float, e.g. 15.0)
      - rem_percentage (float, e.g. 22.0)
      - room_temperature_celsius (float, e.g. 19.5)
      - alcohol_units_before_bed (float, e.g. 0.0)
      - bmi (float, e.g. 24.0)
    """
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    model_path = os.path.join(models_dir, 'ols_model.pkl')
    meta_path = os.path.join(models_dir, 'feature_metadata.json')
    
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
        
    selected_features = metadata['selected_features']
    
    # Preprocess categorical inputs if strings passed
    clean_inputs = user_inputs.copy()
    if isinstance(clean_inputs.get('is_healthy_mental'), str):
        clean_inputs['is_healthy_mental'] = 1.0 if clean_inputs['is_healthy_mental'].lower() == 'healthy' else 0.0
    if isinstance(clean_inputs.get('has_sleep_disorder'), str):
        clean_inputs['has_sleep_disorder'] = 0.0 if clean_inputs['has_sleep_disorder'].lower() == 'healthy' else 1.0
        
    df_in = pd.DataFrame([clean_inputs])[selected_features]
    scaled_vals = scaler.transform(df_in)
    
    scaled_with_const = np.insert(scaled_vals[0], 0, 1.0)
    pred_score = float(np.dot(scaled_with_const, model.params.values))
    
    # Clip to valid score bounds [1.0, 10.0]
    clipped_score = max(1.0, min(10.0, pred_score))
    return round(clipped_score, 2)

if __name__ == '__main__':
    sample = {
        'sleep_duration_hrs': 8.0,
        'wake_episodes_per_night': 1,
        'stress_score': 3.0,
        'is_healthy_mental': 1,
        'has_sleep_disorder': 0,
        'shift_work': 0,
        'sleep_latency_mins': 12,
        'rem_percentage': 24.0,
        'room_temperature_celsius': 19.0,
        'alcohol_units_before_bed': 0,
        'bmi': 22.5
    }
    score = predict_sleep_quality(sample)
    print(f"Sample Prediction -> Sleep Quality Score: {score} / 10.0")
