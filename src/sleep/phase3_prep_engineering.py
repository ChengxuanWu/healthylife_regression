"""
CRISP-DM Phase 3: Data Preparation & Feature Engineering
Sleep Health & Quality Multiple Linear Regression Project
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_phase_3(data_path='sleep_health_dataset.csv'):
    print("=" * 80)
    print("CRISP-DM PHASE 3: DATA PREPARATION & FEATURE ENGINEERING")
    print("=" * 80)
    
    df = pd.read_csv(data_path)
    
    # 3.1 Feature Engineering
    print("\n3.1 Constructing Domain-Specific Engineered Features:")
    
    # 1. Sleep Fragmentation Index (Nocturnal awakenings normalized by sleep duration)
    # Awakenings per hour of sleep. Captures sleep continuity breakdown.
    df['sleep_fragmentation_index'] = df['wake_episodes_per_night'] / (df['sleep_duration_hrs'] + 0.1)
    print("    [Engineered 1] 'sleep_fragmentation_index' = wake_episodes_per_night / (sleep_duration_hrs + 0.1)")
    
    # 2. Restorative Sleep Percentage (Sum of Deep Sleep and REM sleep)
    df['restorative_sleep_pct'] = df['rem_percentage'] + df['deep_sleep_percentage']
    print("    [Engineered 2] 'restorative_sleep_pct' = rem_percentage + deep_sleep_percentage")
    
    # 3. Work-Stress Load Interaction (Hours worked * subjective stress)
    df['work_stress_load'] = df['work_hours_that_day'] * df['stress_score']
    print("    [Engineered 3] 'work_stress_load' = work_hours_that_day * stress_score")
    
    # 4. Clinical Condition Indicators
    df['is_healthy_mental'] = (df['mental_health_condition'] == 'Healthy').astype(float)
    df['has_sleep_disorder'] = (df['sleep_disorder_risk'] != 'Healthy').astype(float)
    print("    [Engineered 4] 'is_healthy_mental' & 'has_sleep_disorder' binary indicators")
    
    # 5. Weekend Sleep Catch-Up / Social Jetlag absolute deviation
    df['social_jetlag_abs'] = df['weekend_sleep_diff_hrs'].abs()
    print("    [Engineered 5] 'social_jetlag_abs' = |weekend_sleep_diff_hrs|")
    
    # 3.2 Candidate Predictor Pool (Excluding downstream post-sleep outcomes: cognitive_performance_score, felt_rested)
    candidate_features = [
        'sleep_duration_hrs',
        'wake_episodes_per_night',
        'stress_score',
        'sleep_fragmentation_index',
        'rem_percentage',
        'deep_sleep_percentage',
        'restorative_sleep_pct',
        'sleep_latency_mins',
        'shift_work',
        'work_hours_that_day',
        'work_stress_load',
        'is_healthy_mental',
        'has_sleep_disorder',
        'room_temperature_celsius',
        'screen_time_before_bed_mins',
        'caffeine_mg_before_bed',
        'alcohol_units_before_bed',
        'heart_rate_resting_bpm',
        'social_jetlag_abs',
        'bmi',
        'age',
        'steps_that_day',
        'exercise_day',
        'sleep_aid_used'
    ]
    
    X = df[candidate_features].copy()
    y = df['sleep_quality_score'].copy()
    
    print(f"\n3.3 Design Matrix: {X.shape[1]} candidate features extracted.")
    
    # 3.4 Leakage-Free Train/Test Split (80% Train, 20% Held-Out Test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"3.4 Partitioning: {len(X_train_raw):,} Training rows (80%), {len(X_test_raw):,} Held-out Test rows (20%)")
    
    # 3.5 Standard Scaling (Fit strictly on Training set, transform on Test set)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_raw),
        columns=candidate_features,
        index=X_train_raw.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=candidate_features,
        index=X_test_raw.index
    )
    print("3.5 Standardization: Applied StandardScaler with zero training leakage.")
    print("=" * 80)
    
    return {
        'df': df,
        'X_train_raw': X_train_raw,
        'X_test_raw': X_test_raw,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'candidate_features': candidate_features
    }

if __name__ == '__main__':
    run_phase_3()
