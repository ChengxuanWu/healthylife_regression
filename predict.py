import joblib
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

def load_artifacts(models_dir='models'):
    scaler = joblib.load(f'{models_dir}/scaler.pkl')
    model = joblib.load(f'{models_dir}/ols_model.pkl')
    with open(f'{models_dir}/feature_metadata.json', 'r') as f:
        metadata = json.load(f)
    return scaler, model, metadata['selected_features']

def predict_life_expectancy(user_inputs: dict, scaler=None, model=None, selected_features=None):
    """
    Predicts national life expectancy (in years) based on user inputs.
    
    user_inputs: dict of raw feature values:
      - adult_mortality (probability per 1,000)
      - under_five_deaths (per 1,000)
      - alcohol (litres of pure alcohol per capita)
      - percentage_expenditure (% of GDP per capita on health)
      - measles (reported cases per 1,000)
      - bmi (average BMI of population)
      - hiv_aids (deaths per 1,000 live births)
      - gdp (GDP per capita in USD)
      - thinness__1_19_years (% thinness among children & adolescents)
      - schooling (years of schooling)
      - income_composition_of_resources (Human Development Index, 0-1)
      - polio (% immunization coverage among 1-year-olds)
      - diphtheria (% immunization coverage among 1-year-olds)
      - hepatitis_b (% immunization coverage among 1-year-olds)
    """
    if scaler is None or model is None or selected_features is None:
        scaler, model, selected_features = load_artifacts()

    input_df = pd.DataFrame([user_inputs])
    
    # 1. Domain feature transformations
    input_df['log_gdp'] = np.log1p(input_df['gdp'])
    input_df['log_hiv'] = np.log1p(input_df['hiv_aids'])
    input_df['human_capital'] = input_df['schooling'] * input_df['income_composition_of_resources']
    input_df['immunization_index'] = (input_df['polio'] + input_df['diphtheria'] + input_df['hepatitis_b']) / 3.0
    
    # Handle column naming differences if any
    if 'thinness_1_19_years' in input_df.columns and 'thinness__1_19_years' not in input_df.columns:
        input_df['thinness__1_19_years'] = input_df['thinness_1_19_years']

    # 2. Extract selected features in expected order
    input_filtered = input_df[selected_features]
    
    # 3. Scale features using the production scaler
    input_scaled = pd.DataFrame(scaler.transform(input_filtered), columns=selected_features)
    
    # 4. Add constant for OLS intercept
    input_scaled_with_const = sm.add_constant(input_scaled, has_constant='add')
    
    # 5. Model prediction
    prediction = model.predict(input_scaled_with_const)
    return round(float(prediction.iloc[0]), 2)

if __name__ == '__main__':
    # Test cases: Developed country profile vs Developing country profile
    developed_profile = {
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
    
    developing_profile = {
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

    pred_dev = predict_life_expectancy(developed_profile)
    pred_ing = predict_life_expectancy(developing_profile)
    
    print("--- Inference Test Results ---")
    print(f"Predicted Life Expectancy for Developed Nation Profile:  {pred_dev} years")
    print(f"Predicted Life Expectancy for Developing Nation Profile: {pred_ing} years")
