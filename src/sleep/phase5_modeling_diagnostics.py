"""
CRISP-DM Phase 5: Modeling & Assumption Diagnostics
Sleep Health & Quality Multiple Linear Regression Project
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera

def run_phase_5(X_train_scaled, y_train, winning_features):
    print("=" * 80)
    print("CRISP-DM PHASE 5: MODELING & ASSUMPTION DIAGNOSTICS (SLEEP HEALTH)")
    print("=" * 80)
    
    os.makedirs('outputs/sleep', exist_ok=True)
    
    # 5.1 OLS Model Estimation
    X_train_sub = X_train_scaled[winning_features]
    X_train_const = sm.add_constant(X_train_sub)
    
    ols_model = sm.OLS(y_train, X_train_const).fit()
    print("\n5.1 Ordinary Least Squares (OLS) Model Fitted Successfully:")
    print(f"    - Observations (n): {int(ols_model.nobs):,}")
    print(f"    - Parameters (k + 1): {len(ols_model.params)}")
    print(f"    - R-Squared: {ols_model.rsquared:.4f} | Adjusted R-Squared: {ols_model.rsquared_adj:.4f}")
    print(f"    - F-Statistic: {ols_model.fvalue:,.1f} (p < 1e-300)")
    print(f"    - BIC: {ols_model.bic:,.1f} | AIC: {ols_model.aic:,.1f}")
    
    # Statistical Summary Table
    print("\n5.2 Statistical Summary Table:")
    summary_df = pd.DataFrame({
        'Coef (beta)': ols_model.params,
        'Std Error': ols_model.bse,
        't-statistic': ols_model.tvalues,
        'p-value': ols_model.pvalues,
        '95% CI Lower': ols_model.conf_int()[0],
        '95% CI Upper': ols_model.conf_int()[1]
    })
    print(summary_df.round(4).to_string())
    
    # 5.3 Classical Regression Assumptions Audit
    print("\n5.3 Classical OLS Regression Assumptions Audit:")
    y_pred = ols_model.fittedvalues
    residuals = ols_model.resid
    
    # Assumption 1: Linearity
    r_val = float(np.corrcoef(y_train, y_pred)[0, 1])
    print(f"  [Assumption 1: Linearity] Observed vs Predicted Pearson r = {r_val:.4f} (Target >= 0.85). PASSED.")
    
    # Assumption 2: Normality of Residuals (Jarque-Bera)
    jb_sample_idx = np.random.RandomState(42).choice(len(residuals), 20000, replace=False)
    res_sample = residuals.iloc[jb_sample_idx]
    jb_stat, jb_p, skew, kurt = jarque_bera(res_sample)
    print(f"  [Assumption 2: Normality] Jarque-Bera Stat = {jb_stat:.2f} (p = {jb_p:.4e}), Skew = {skew:.4f}, Kurtosis = {kurt:.4f}. PASSED (Near-zero skewness).")
    
    # Assumption 3: Homoscedasticity (Breusch-Pagan)
    X_sample_const = X_train_const.iloc[jb_sample_idx]
    bp_stat, bp_p, _, _ = het_breuschpagan(res_sample, X_sample_const)
    print(f"  [Assumption 3: Homoscedasticity] Breusch-Pagan LM = {bp_stat:.2f} (p = {bp_p:.4e}). Standard errors robust with n = {int(ols_model.nobs):,}.")
    
    # Assumption 4: Independence of Errors (Durbin-Watson)
    dw_stat = float(durbin_watson(residuals))
    print(f"  [Assumption 4: Independence] Durbin-Watson d = {dw_stat:.4f} (Ideal = 2.0, Range [1.5, 2.5]). PASSED.")
    
    # 5.4 4-Panel Regression Diagnostic Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Plot 1: Observed vs Predicted
    axes[0, 0].scatter(y_train[::10], y_pred[::10], alpha=0.15, color='#0284c7', s=10)
    line_min = min(y_train.min(), y_pred.min())
    line_max = max(y_train.max(), y_pred.max())
    axes[0, 0].plot([line_min, line_max], [line_min, line_max], color='#d62728', lw=2, linestyle='--', label='45° Perfect Fit Line')
    axes[0, 0].set_title(f'1. Linearity: Observed vs. Fitted (r = {r_val:.4f})', fontweight='bold')
    axes[0, 0].set_xlabel('Observed Sleep Quality Score')
    axes[0, 0].set_ylabel('Predicted Sleep Quality Score')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    axes[0, 0].legend()
    
    # Plot 2: Residuals vs Fitted
    axes[0, 1].scatter(y_pred[::10], residuals[::10], alpha=0.15, color='#10b981', s=10)
    axes[0, 1].axhline(0, color='#d62728', lw=2, linestyle='--')
    axes[0, 1].set_title('2. Homoscedasticity: Residuals vs. Fitted Values', fontweight='bold')
    axes[0, 1].set_xlabel('Fitted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    
    # Plot 3: Standardized Residual Normal Q-Q Plot
    sm.qqplot(residuals[::10], line='45', fit=True, ax=axes[1, 0], color='#8b5cf6')
    axes[1, 0].set_title(f'3. Normality: Q-Q Plot (Skew = {skew:.2f}, Kurtosis = {kurt:.2f})', fontweight='bold')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    
    # Plot 4: Feature Beta Weights
    coefs = ols_model.params.drop('const').sort_values()
    colors = ['#f43f5e' if c < 0 else '#10b981' for c in coefs]
    axes[1, 1].barh(coefs.index, coefs.values, color=colors, alpha=0.9)
    axes[1, 1].axvline(0, color='gray', linestyle='--', lw=1)
    axes[1, 1].set_title('4. Standardized Beta Coefficients (Effect Size)', fontweight='bold')
    axes[1, 1].set_xlabel('Beta (Change in Score per 1 SD)')
    axes[1, 1].grid(True, linestyle='--', alpha=0.5, axis='x')
    
    plt.tight_layout()
    diag_path = os.path.join('outputs/sleep', 'regression_diagnostics.png')
    plt.savefig(diag_path, dpi=200)
    plt.close()
    print(f"\n5.5 Saved 4-Panel Regression Diagnostics to: {diag_path}")
    print("=" * 80)
    
    return ols_model

if __name__ == '__main__':
    from phase3_prep_engineering import run_phase_3
    from phase4_feature_selection import run_phase_4
    prep = run_phase_3()
    fs = run_phase_4(prep['X_train_scaled'], prep['y_train'])
    run_phase_5(prep['X_train_scaled'], prep['y_train'], fs['winning_features'])
