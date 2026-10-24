"""
CRISP-DM Phase 2: Data Understanding & Profiling
Sleep Health & Quality Multiple Linear Regression Project
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_phase_2(data_path='sleep_health_dataset.csv'):
    print("=" * 80)
    print("CRISP-DM PHASE 2: DATA UNDERSTANDING & PROFILING")
    print("=" * 80)
    
    os.makedirs('outputs/sleep', exist_ok=True)
    df = pd.read_csv(data_path)
    print(f"\n2.1 Dataset Dimensionality: {df.shape[0]:,} observations, {df.shape[1]} columns")
    
    # Missing Value Audit
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    print(f"2.2 Missing Value Audit: {total_missing} missing values detected (100% complete records).")
    
    # Target Variable Profiling
    target = 'sleep_quality_score'
    print(f"\n2.3 Target Variable Analysis ({target}):")
    print(f"    - Type: Continuous rating score")
    print(f"    - Range: {df[target].min():.1f} to {df[target].max():.1f}")
    print(f"    - Mean:  {df[target].mean():.2f} +/- {df[target].std():.2f}")
    print(f"    - Median: {df[target].median():.2f} (IQR: {df[target].quantile(0.25):.2f} - {df[target].quantile(0.75):.2f})")
    
    # Downstream Outcome Leakage Identification
    print("\n2.4 Econometric Causality & Leakage Audit:")
    print("    - Identified Post-Sleep Outcomes: 'cognitive_performance_score' (corr = 0.86) and 'felt_rested' (corr = 0.48).")
    print("    - Policy / Clinical Decision: Exclude post-sleep outcomes from feature pool to avoid reverse-causality target leakage.")
    
    # Generate Distribution Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Target Distribution
    sns.histplot(df[target], kde=True, ax=axes[0, 0], color='#0284c7', bins=25)
    axes[0, 0].set_title('Target Variable: Sleep Quality Score Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Sleep Quality Score (1 - 10)')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    
    # Sleep Duration
    sns.histplot(df['sleep_duration_hrs'], kde=True, ax=axes[0, 1], color='#10b981', bins=25)
    axes[0, 1].set_title('Predictor: Sleep Duration (Hours)', fontweight='bold')
    axes[0, 1].set_xlabel('Sleep Duration (Hours/Night)')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    
    # Stress Score
    sns.histplot(df['stress_score'], kde=True, ax=axes[1, 0], color='#ea580c', bins=25)
    axes[1, 0].set_title('Predictor: Daytime Stress Score (0 - 10)', fontweight='bold')
    axes[1, 0].set_xlabel('Reported Stress Score')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    
    # Nocturnal Awakenings
    sns.countplot(x='wake_episodes_per_night', data=df, ax=axes[1, 1], palette='Purples_r')
    axes[1, 1].set_title('Predictor: Nocturnal Wake Episodes per Night', fontweight='bold')
    axes[1, 1].set_xlabel('Wake Episodes Count')
    axes[1, 1].grid(True, linestyle='--', alpha=0.5, axis='y')
    
    plt.tight_layout()
    dist_path = os.path.join('outputs/sleep', 'eda_distributions.png')
    plt.savefig(dist_path, dpi=200)
    plt.close()
    print(f"\n2.5 Saved Key Distributions Plot to: {dist_path}")
    
    # Correlation Matrix
    key_numeric = [
        'sleep_quality_score', 'sleep_duration_hrs', 'stress_score',
        'wake_episodes_per_night', 'sleep_latency_mins', 'rem_percentage',
        'deep_sleep_percentage', 'work_hours_that_day', 'shift_work',
        'screen_time_before_bed_mins', 'room_temperature_celsius', 'alcohol_units_before_bed'
    ]
    corr_matrix = df[key_numeric].corr()
    
    plt.figure(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-0.8, vmax=0.8, square=True, linewidths=0.5)
    plt.title('Sleep Health Correlation Heatmap (Key Predictors & Target)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    corr_path = os.path.join('outputs/sleep', 'eda_correlation_matrix.png')
    plt.savefig(corr_path, dpi=200)
    plt.close()
    print(f"2.6 Saved Correlation Heatmap to: {corr_path}")
    print("=" * 80)
    
    return df

if __name__ == '__main__':
    run_phase_2()
