"""
CRISP-DM Phase 3: Data Preparation & Feature Engineering
WHO Life Expectancy Multiple Linear Regression Project
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_phase_3(data_path='Life Expectancy Data.csv'):
    print("=" * 80)
    print("CRISP-DM PHASE 3: DATA PREPARATION & FEATURE ENGINEERING")
    print("=" * 80)
    
    # 1. Clean Column Headers
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_').str.replace('-', '_')
    print(f"3.1 Raw observations: {len(df)}")
    
    # Drop rows missing target
    df = df.dropna(subset=['life_expectancy']).reset_index(drop=True)
    print(f"    Observations after dropping missing target: {len(df)}")
    
    # 2. Impute continuous columns by status group median
    continuous_cols = df.select_dtypes(include=[np.number]).columns
    for col in continuous_cols:
        if col != 'life_expectancy':
            df[col] = df.groupby('status')[col].transform(lambda x: x.fillna(x.median()))
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
    print("    [OK] Group-median imputation completed (partitioned by status).")
    
    # 3. Domain-Specific Feature Engineering
    print("\n3.2 Applying Domain Feature Engineering:")
    # Composite Immunization Index
    df['immunization_index'] = (df['polio'] + df['diphtheria'] + df['hepatitis_b']) / 3.0
    print("    - immunization_index = (polio + diphtheria + hepatitis_b) / 3")
    
    # Human Capital Interaction
    df['human_capital'] = df['schooling'] * df['income_composition_of_resources']
    print("    - human_capital = schooling * income_composition_of_resources")
    
    # Log Transforms
    df['log_gdp'] = np.log1p(df['gdp'])
    df['log_hiv'] = np.log1p(df['hiv_aids'])
    print("    - log_gdp = ln(1 + gdp)")
    print("    - log_hiv = ln(1 + hiv_aids)")

    # 4. Construct Design Matrices
    # Curated candidate pool: replacing redundant collinear pairs with engineered terms
    curated_candidates = [
        'adult_mortality', 'under_five_deaths', 'alcohol', 'percentage_expenditure',
        'measles', 'bmi', 'total_expenditure', 'log_hiv', 'log_gdp', 'population',
        'thinness__1_19_years', 'human_capital', 'immunization_index', 'status'
    ]
    
    X_cur = pd.get_dummies(df[curated_candidates], columns=['status'], drop_first=True)
    y = df['life_expectancy']
    
    # 5. Train/Test Split (80/20, fixed seed)
    X_train, X_test, y_train, y_test = train_test_split(X_cur, y, test_size=0.20, random_state=42)
    print(f"\n3.3 Partitioning Dataset:")
    print(f"    - Training Set: {X_train.shape[0]} rows (80%)")
    print(f"    - Test Set:     {X_test.shape[0]} rows (20%)")
    
    # 6. Leakage-Free Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    print("    [OK] StandardScaler fit strictly on Training Set; Test Set transformed.")
    print("=" * 80)
    
    return {
        'df': df,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'X_cur': X_cur
    }

if __name__ == '__main__':
    run_phase_3()
