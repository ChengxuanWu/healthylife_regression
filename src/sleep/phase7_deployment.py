"""
CRISP-DM Phase 7: Deployment & Delivery
Sleep Health & Quality Multiple Linear Regression Project
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

def run_phase_7(winning_model, winning_features, df, eval_df):
    print("=" * 80)
    print("CRISP-DM PHASE 7: DEPLOYMENT & DELIVERY (SLEEP HEALTH)")
    print("=" * 80)
    
    os.makedirs('models/sleep', exist_ok=True)
    
    # Fit dedicated production scaler strictly on winning features
    prod_scaler = StandardScaler()
    X_prod = df[winning_features]
    prod_scaler.fit(X_prod)
    
    # Fit final production OLS model on entire dataset for maximum precision
    X_prod_scaled = pd.DataFrame(prod_scaler.transform(X_prod), columns=winning_features)
    X_prod_const = sm.add_constant(X_prod_scaled)
    prod_model = sm.OLS(df['sleep_quality_score'], X_prod_const).fit()
    
    # 7.1 Serialize Models & Scalers
    scaler_path = 'models/sleep/scaler.pkl'
    model_path = 'models/sleep/ols_model.pkl'
    meta_path = 'models/sleep/feature_metadata.json'
    
    joblib.dump(prod_scaler, scaler_path)
    joblib.dump(prod_model, model_path)
    print(f"7.1 Exported Production Scaler: {scaler_path}")
    print(f"    Exported Production OLS Model: {model_path}")
    
    # 7.2 Export Metadata JSON
    metadata = {
        'target_variable': 'sleep_quality_score',
        'units': 'rating (1.0 to 10.0)',
        'selected_features': winning_features,
        'coefficients': {k: float(v) for k, v in prod_model.params.to_dict().items()},
        'pvalues': {k: float(v) for k, v in prod_model.pvalues.to_dict().items()},
        'r_squared': round(float(prod_model.rsquared), 4),
        'adj_r_squared': round(float(prod_model.rsquared_adj), 4),
        'f_statistic': round(float(prod_model.fvalue), 1),
        'model_architecture_comparison': eval_df.to_dict(orient='records')
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"7.2 Exported Feature Metadata: {meta_path}")
    
    # 7.3 Validation Inference Test
    sample_raw = {
        'sleep_duration_hrs': 7.5,
        'wake_episodes_per_night': 2.0,
        'stress_score': 4.0,
        'is_healthy_mental': 1.0,
        'has_sleep_disorder': 0.0,
        'shift_work': 0.0,
        'sleep_latency_mins': 15.0,
        'rem_percentage': 22.0,
        'room_temperature_celsius': 19.5,
        'alcohol_units_before_bed': 0.0,
        'bmi': 23.5
    }
    
    input_df = pd.DataFrame([sample_raw])[winning_features]
    scaled_vals = prod_scaler.transform(input_df)
    scaled_with_const = np.insert(scaled_vals[0], 0, 1.0)
    pred_score = float(np.dot(scaled_with_const, prod_model.params.values))
    
    print(f"\n7.4 Deployment Inference Sanity Check:")
    print(f"    Input: 7.5h sleep, 2 wakes, stress 4.0, healthy mental, no shift work, 19.5C room")
    print(f"    Predicted Sleep Quality Score: {pred_score:.2f} / 10.0 (Baseline: {prod_model.params['const']:.2f})")
    print("=" * 80)
    
    return metadata

if __name__ == '__main__':
    from phase3_prep_engineering import run_phase_3
    from phase4_feature_selection import run_phase_4
    from phase6_evaluation_insights import run_phase_6
    
    prep = run_phase_3()
    fs = run_phase_4(prep['X_train_scaled'], prep['y_train'])
    eval_res = run_phase_6(
        prep['X_train_scaled'], prep['X_test_scaled'],
        prep['y_train'], prep['y_test'],
        fs['vif_features'], fs['lasso_features'], fs['winning_features']
    )
    run_phase_7(eval_res['winning_model'], fs['winning_features'], prep['df'], eval_res['eval_df'])
