"""
CRISP-DM Phase 7: Deployment & Delivery
WHO Life Expectancy Multiple Linear Regression Project
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

def save_production_artifacts(X_cur, winning_features, winning_model, eval_df, models_dir='models'):
    """
    Serializes models and scalers for zero-dependency inference.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Fit scaler on raw unscaled winning features
    prod_scaler = StandardScaler()
    prod_scaler.fit(X_cur[winning_features])
    
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    model_path = os.path.join(models_dir, 'ols_model.pkl')
    meta_path = os.path.join(models_dir, 'feature_metadata.json')
    
    joblib.dump(prod_scaler, scaler_path)
    joblib.dump(winning_model, model_path)
    
    metadata = {
        'target_variable': 'life_expectancy',
        'units': 'years',
        'selected_features': winning_features,
        'coefficients': winning_model.params.to_dict(),
        'pvalues': winning_model.pvalues.to_dict(),
        'model_architecture_comparison': eval_df.to_dict(orient='records')
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"  [OK] Saved Scaler:    {scaler_path}")
    print(f"  [OK] Saved OLS Model: {model_path}")
    print(f"  [OK] Saved Metadata:  {meta_path}")
    return prod_scaler

def predict_life_expectancy(user_inputs: dict, models_dir='models'):
    """
    Production inference function.
    user_inputs: dict of raw feature values (e.g. {'schooling': 14, 'alcohol': 3.2, ...})
    Returns: Predicted life expectancy in years (rounded to 2 decimal places).
    """
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    model = joblib.load(os.path.join(models_dir, 'ols_model.pkl'))
    with open(os.path.join(models_dir, 'feature_metadata.json'), 'r') as f:
        meta = json.load(f)
    selected_features = meta['selected_features']
    
    input_df = pd.DataFrame([user_inputs])
    
    # 1. Apply Feature Engineering Transformations
    input_df['log_gdp'] = np.log1p(input_df['gdp'])
    input_df['log_hiv'] = np.log1p(input_df['hiv_aids'])
    input_df['human_capital'] = input_df['schooling'] * input_df['income_composition_of_resources']
    input_df['immunization_index'] = (input_df['polio'] + input_df['diphtheria'] + input_df['hepatitis_b']) / 3.0
    
    if 'thinness_1_19_years' in input_df.columns and 'thinness__1_19_years' not in input_df.columns:
        input_df['thinness__1_19_years'] = input_df['thinness_1_19_years']
        
    # 2. Select Features, Scale, and Predict
    input_filtered = input_df[selected_features]
    input_scaled = pd.DataFrame(scaler.transform(input_filtered), columns=selected_features)
    input_scaled_with_const = sm.add_constant(input_scaled, has_constant='add')
    
    prediction = model.predict(input_scaled_with_const)
    return round(float(prediction.iloc[0]), 2)

def run_phase_7(X_cur, winning_features, winning_model, eval_df, models_dir='models'):
    print("=" * 80)
    print("CRISP-DM PHASE 7: DEPLOYMENT & DELIVERY")
    print("=" * 80)
    
    print("\n7.1 Serializing Production Artifacts:")
    save_production_artifacts(X_cur, winning_features, winning_model, eval_df, models_dir=models_dir)
    
    print("\n7.2 Running Interactive 'What-If' Longevity Scoring Engine:")
    # Archetype 1: Developed Nation
    profile_developed = {
        'adult_mortality': 75.0,
        'under_five_deaths': 4.0,
        'alcohol': 9.8,
        'percentage_expenditure': 4500.0,
        'measles': 0.0,
        'bmi': 26.5,
        'hiv_aids': 0.1,
        'gdp': 42000.0,
        'thinness__1_19_years': 0.8,
        'schooling': 16.2,
        'income_composition_of_resources': 0.91,
        'polio': 96.0,
        'diphtheria': 96.0,
        'hepatitis_b': 94.0
    }
    
    # Archetype 2: Developing Nation
    profile_developing = {
        'adult_mortality': 260.0,
        'under_five_deaths': 65.0,
        'alcohol': 1.5,
        'percentage_expenditure': 80.0,
        'measles': 500.0,
        'bmi': 19.0,
        'hiv_aids': 2.5,
        'gdp': 900.0,
        'thinness__1_19_years': 8.5,
        'schooling': 8.0,
        'income_composition_of_resources': 0.45,
        'polio': 65.0,
        'diphtheria': 68.0,
        'hepatitis_b': 62.0
    }
    
    pred_dev = predict_life_expectancy(profile_developed, models_dir=models_dir)
    pred_ing = predict_life_expectancy(profile_developing, models_dir=models_dir)
    
    print(f"    [Scenario A] Developed Nation Profile:  {pred_dev:5.2f} years")
    print(f"    [Scenario B] Developing Nation Profile: {pred_ing:5.2f} years")
    print(f"    -> Longevity Gap: {pred_dev - pred_ing:5.2f} years")
    print("=" * 80)
    return True

if __name__ == '__main__':
    from phase3_prep_engineering import run_phase_3
    from phase4_feature_selection import run_phase_4
    from phase5_modeling_diagnostics import run_phase_5
    from phase6_evaluation_insights import run_phase_6
    
    prep = run_phase_3()
    fs = run_phase_4(prep['X_train_scaled'], prep['y_train'])
    m5 = run_phase_5(prep['X_train_scaled'], prep['y_train'], fs['winning_features'])
    e6 = run_phase_6(prep['df'], prep['X_train_scaled'], prep['X_test_scaled'],
                     prep['y_train'], prep['y_test'], fs['vif_features'],
                     fs['lasso_features'], fs['rfecv_features'])
    run_phase_7(prep['X_cur'], fs['winning_features'], m5['model'], e6['eval_df'])
