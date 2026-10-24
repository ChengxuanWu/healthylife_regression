"""
CRISP-DM Phase 2: Data Understanding & Profiling
WHO Life Expectancy Multiple Linear Regression Project
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_phase_2(data_path='Life Expectancy Data.csv'):
    print("=" * 80)
    print("CRISP-DM PHASE 2: DATA UNDERSTANDING & PROFILING")
    print("=" * 80)
    
    os.makedirs('outputs', exist_ok=True)
    
    # 2.1 Initial Inspection
    df_raw = pd.read_csv(data_path)
    print(f"\n2.1 Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    print("Raw Column Headers (inspecting trailing/leading whitespace):")
    for col in df_raw.columns:
        has_space = col.startswith(' ') or col.endswith(' ') or '  ' in col
        flag = " [!] Needs trimming" if has_space else " [OK]"
        print(f"  - '{col}'{flag}")
        
    # Standardize column headers for analysis
    df = df_raw.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_').str.replace('-', '_')
    
    # Target variable profiling
    target = 'life_expectancy'
    missing_target = df[target].isnull().sum()
    print(f"\nTarget Variable '{target}':")
    print(f"  Missing values: {missing_target} rows ({missing_target/len(df)*100:.2f}%)")
    target_stats = df[target].describe()
    print(f"  Range: [{target_stats['min']:.1f}, {target_stats['max']:.1f}] years | Mean: {target_stats['mean']:.2f} | Std: {target_stats['std']:.2f}")

    # 2.2 Missing Values Audit
    print("\n2.2 Missing Values Audit Across Features:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing_Count': missing, 'Missing_Pct': missing_pct})
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(by='Missing_Pct', ascending=False)
    for idx, row in missing_df.iterrows():
        print(f"  - {idx:32s}: {int(row['Missing_Count']):4d} missing ({row['Missing_Pct']:5.2f}%)")
        
    # Plot missing values
    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing_df['Missing_Pct'], y=missing_df.index, palette='Reds_r')
    plt.title('Percentage of Missing Values per Feature', fontsize=12, fontweight='bold')
    plt.xlabel('Missing Percentage (%)')
    plt.ylabel('Feature')
    plt.tight_layout()
    missing_plot_path = os.path.join('outputs', 'eda_missingness.png')
    plt.savefig(missing_plot_path, dpi=200)
    plt.close()
    print(f"  -> Saved missingness chart to: {missing_plot_path}")

    # 2.3 Known Quirks: Multicollinearity & Skewness
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    print("\n2.3 Severe Multicollinearity Pairs Detected:")
    pairs_to_check = [
        ('infant_deaths', 'under_five_deaths'),
        ('thinness__1_19_years', 'thinness_5_9_years'),
        ('gdp', 'percentage_expenditure'),
        ('polio', 'diphtheria'),
        ('schooling', 'income_composition_of_resources')
    ]
    for col1, col2 in pairs_to_check:
        if col1 in corr_matrix.columns and col2 in corr_matrix.columns:
            r = corr_matrix.loc[col1, col2]
            risk = "[HIGH RISK]" if abs(r) > 0.80 else "[MODERATE]"
            print(f"  - {col1} <-> {col2}: Pearson r = {r:.4f} {risk}")

    # Right skewness profiling
    print("\nRight-Skewness Profiling (Fisher-Pearson Skewness coefficient):")
    skew_cols = ['gdp', 'population', 'hiv_aids', 'measles']
    for col in skew_cols:
        if col in numeric_df.columns:
            skew_val = numeric_df[col].skew()
            print(f"  - {col:15s}: Skew = {skew_val:6.2f} (Extreme right tail requires log transform)")

    # Correlation Heatmap
    plt.figure(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, mask=mask, linewidths=0.5)
    plt.title('Correlation Heatmap of Raw WHO Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    corr_plot_path = os.path.join('outputs', 'eda_correlation_matrix.png')
    plt.savefig(corr_plot_path, dpi=200)
    plt.close()
    print(f"  -> Saved correlation heatmap to: {corr_plot_path}")
    print("=" * 80)
    return df

if __name__ == '__main__':
    run_phase_2()
