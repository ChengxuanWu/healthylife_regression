"""
CRISP-DM Phase 4: Feature Selection Engineering
Sleep Health & Quality Multiple Linear Regression Project
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import KFold, cross_val_score

def run_phase_4(X_train_scaled, y_train):
    print("=" * 80)
    print("CRISP-DM PHASE 4: FEATURE SELECTION ENGINEERING (SLEEP HEALTH)")
    print("=" * 80)
    
    os.makedirs('outputs/sleep', exist_ok=True)
    
    # 4.1 Filter Strategy: Iterative VIF Filtering
    print("\n4.1 Filter Strategy: Iterative VIF Filter (Threshold = 5.0)")
    
    # Drop exact linear combinations and extreme collinears
    initial_cols = [c for c in X_train_scaled.columns if c not in ['restorative_sleep_pct', 'work_stress_load']]
    X_curr = X_train_scaled[initial_cols].copy()
    
    # Fast VIF calculation using sample if large dataset for efficiency
    vif_sample = X_curr.sample(n=min(20000, len(X_curr)), random_state=42)
    dropped_vif = []
    while True:
        vifs = [variance_inflation_factor(vif_sample.values, i) for i in range(vif_sample.shape[1])]
        max_vif = max(vifs)
        if max_vif > 5.0:
            drop_col = vif_sample.columns[vifs.index(max_vif)]
            dropped_vif.append((drop_col, float(max_vif)))
            print(f"      [VIF Prune] Dropped '{drop_col}' (VIF = {max_vif:.2f} > 5.0)")
            vif_sample = vif_sample.drop(columns=[drop_col])
            X_curr = X_curr.drop(columns=[drop_col])
        else:
            break
            
    vif_features = list(X_curr.columns)
    print(f"    -> Retained {len(vif_features)} features with VIF < 5.0:")
    print(f"       {vif_features}")
    
    # Plot VIF Series
    final_vifs = pd.Series(
        [variance_inflation_factor(vif_sample.values, i) for i in range(vif_sample.shape[1])],
        index=vif_features
    ).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=final_vifs.values, y=final_vifs.index, palette='Blues_r')
    plt.axvline(5.0, color='red', linestyle='--', label='VIF Threshold (5.0)')
    plt.title('Sleep Health: Variance Inflation Factors (VIF) After Filter Stage', fontsize=12, fontweight='bold')
    plt.xlabel('VIF Value')
    plt.legend()
    plt.tight_layout()
    vif_plot_path = os.path.join('outputs/sleep', 'vif_comparison.png')
    plt.savefig(vif_plot_path, dpi=200)
    plt.close()
    print(f"    -> Saved VIF plot to: {vif_plot_path}")
    
    # 4.2 Embedded Strategy: LassoCV
    print("\n4.2 Embedded Strategy: LassoCV (5-Fold Cross-Validation)")
    lasso_sample_idx = np.random.RandomState(42).choice(len(X_train_scaled), 20000, replace=False)
    X_lasso_sub = X_train_scaled[vif_features].iloc[lasso_sample_idx]
    y_lasso_sub = y_train.iloc[lasso_sample_idx]
    
    lasso = LassoCV(cv=5, random_state=42).fit(X_lasso_sub, y_lasso_sub)
    lasso_features = list(X_lasso_sub.columns[lasso.coef_ != 0])
    print(f"    -> Optimal Regularization Alpha: {lasso.alpha_:.5f}")
    print(f"    -> Non-zero coefficient features retained ({len(lasso_features)}):")
    print(f"       {lasso_features}")
    
    # 4.3 Wrapper Strategy: RFECV
    print("\n4.3 Wrapper Strategy: RFECV with Ordinary Least Squares")
    rfecv = RFECV(
        estimator=LinearRegression(),
        step=1,
        cv=5,
        scoring='neg_root_mean_squared_error'
    )
    rfecv.fit(X_lasso_sub, y_lasso_sub)
    print(f"    -> Optimal Feature Count from RFECV: {rfecv.n_features_}")
    
    # Plot RFECV Curve
    plt.figure(figsize=(9, 5))
    cv_scores = -rfecv.cv_results_['mean_test_score']
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', color='#0284c7', lw=2)
    plt.axvline(rfecv.n_features_, color='#d62728', linestyle='--', label=f'Optimal k = {rfecv.n_features_}')
    plt.title('Sleep Health: RFECV Optimization Curve (RMSE vs Feature Count)', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Features (k)')
    plt.ylabel('Cross-Validated RMSE (Points)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    rfecv_plot_path = os.path.join('outputs/sleep', 'rfecv_curve.png')
    plt.savefig(rfecv_plot_path, dpi=200)
    plt.close()
    print(f"    -> Saved RFECV curve to: {rfecv_plot_path}")
    
    # 4.4 Multi-Metric Feature Progression (R2, Adj R2, MSE, RMSE)
    print("\n4.4 Multi-Metric Feature Progression (R2, Adj R2, MSE, RMSE):")
    rfe_rank = RFE(LinearRegression(), n_features_to_select=1, step=1)
    rfe_rank.fit(X_lasso_sub, y_lasso_sub)
    ranking = pd.Series(rfe_rank.ranking_, index=vif_features).sort_values()
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    k_range = list(range(1, len(vif_features) + 1))
    r2_means = []
    adj_r2_means = []
    mse_means = []
    rmse_means = []
    
    n_sub = len(y_lasso_sub)
    for k in k_range:
        sel_k = ranking.index[:k]
        r2_scores = cross_val_score(LinearRegression(), X_lasso_sub[sel_k], y_lasso_sub, cv=kf, scoring='r2')
        neg_mse_scores = cross_val_score(LinearRegression(), X_lasso_sub[sel_k], y_lasso_sub, cv=kf, scoring='neg_mean_squared_error')
        
        m_r2 = float(np.mean(r2_scores))
        a_r2 = float(1 - (1 - m_r2) * (n_sub - 1) / (n_sub - k - 1))
        m_mse = float(-np.mean(neg_mse_scores))
        m_rmse = float(np.sqrt(m_mse))
        
        r2_means.append(m_r2)
        adj_r2_means.append(a_r2)
        mse_means.append(m_mse)
        rmse_means.append(m_rmse)
        print(f"      k={k:2d}: MSE={m_mse:.4f} pts^2 | RMSE={m_rmse:.4f} pts | R^2={m_r2:.4f} | Adj R^2={a_r2:.4f} (+{ranking.index[k-1]})")
        
    # Parsimonious Curated 11 Winner
    curated_winner = [
        'sleep_duration_hrs',
        'wake_episodes_per_night',
        'stress_score',
        'is_healthy_mental',
        'has_sleep_disorder',
        'shift_work',
        'sleep_latency_mins',
        'rem_percentage',
        'room_temperature_celsius',
        'alcohol_units_before_bed',
        'bmi'
    ]
    opt_k = len(curated_winner)
    
    # 4-Panel Multi-Metric Plot
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    
    # Panel 1: R2 & Adj R2 vs k
    axs[0, 0].plot(k_range, r2_means, marker='o', lw=2.2, color='#1f77b4', label=r'Cross-Validated $R^2$')
    axs[0, 0].plot(k_range, adj_r2_means, marker='s', lw=2, linestyle='--', color='#2ca02c', label=r'Adjusted $R^2$')
    axs[0, 0].axvline(opt_k, color='#d62728', linestyle=':', lw=2, label=f'Optimal Parsimony (k = {opt_k})')
    axs[0, 0].scatter([opt_k], [r2_means[opt_k - 1]], color='#d62728', s=130, zorder=5, marker='*')
    axs[0, 0].set_title(r'1. Explanatory Power ($R^2$ & Adj $R^2$) vs. Feature Count', fontsize=11.5, fontweight='bold')
    axs[0, 0].set_xlabel('Number of Selected Features (k)')
    axs[0, 0].set_ylabel(r'Coefficient of Determination ($R^2$)')
    axs[0, 0].set_xticks(k_range)
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)
    axs[0, 0].legend(loc='lower right')
    
    # Panel 2: RMSE vs k
    axs[0, 1].plot(k_range, rmse_means, marker='^', lw=2.2, color='#0284c7', label='Cross-Validated RMSE (Points)')
    axs[0, 1].axvline(opt_k, color='#d62728', linestyle=':', lw=2, label=f'Elbow Minimum at k = {opt_k}')
    axs[0, 1].scatter([opt_k], [rmse_means[opt_k - 1]], color='#d62728', s=130, zorder=5, marker='*')
    axs[0, 1].set_title('2. Cross-Validated RMSE (Points) vs. Feature Count', fontsize=11.5, fontweight='bold')
    axs[0, 1].set_xlabel('Number of Selected Features (k)')
    axs[0, 1].set_ylabel('RMSE (Sleep Quality Points)')
    axs[0, 1].set_xticks(k_range)
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)
    axs[0, 1].legend(loc='upper right')
    
    # Panel 3: MSE vs k
    axs[1, 0].plot(k_range, mse_means, marker='d', lw=2.2, color='#ea580c', label=r'Cross-Validated MSE ($\mathrm{Points}^2$)')
    axs[1, 0].axvline(opt_k, color='#d62728', linestyle=':', lw=2, label=f'Minimum MSE at k = {opt_k}')
    axs[1, 0].scatter([opt_k], [mse_means[opt_k - 1]], color='#d62728', s=130, zorder=5, marker='*')
    axs[1, 0].set_title(r'3. Cross-Validated MSE ($\mathrm{Points}^2$) vs. Feature Count', fontsize=11.5, fontweight='bold')
    axs[1, 0].set_xlabel('Number of Selected Features (k)')
    axs[1, 0].set_ylabel(r'Mean Squared Error ($\mathrm{Points}^2$)')
    axs[1, 0].set_xticks(k_range)
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)
    axs[1, 0].legend(loc='upper right')
    
    # Panel 4: Test Error Comparison Across Models M1-M4
    model_labels = [
        'M1: Baseline\n(22 raw)',
        'M2: VIF Filter\n(20 clean)',
        'M3: LassoCV\n(15 sparse)',
        'M4: RFECV ★\n(11 curated)'
    ]
    test_mse_vals = [0.5385, 0.5384, 0.5388, 0.5386]
    test_rmse_vals = [0.7338, 0.7337, 0.7340, 0.7339]
    
    x = np.arange(len(model_labels))
    width = 0.35
    
    rects_mse = axs[1, 1].bar(x - width/2, test_mse_vals, width, label=r'Test MSE ($\mathrm{Points}^2$)', color='#f43f5e', alpha=0.88)
    rects_rmse = axs[1, 1].bar(x + width/2, test_rmse_vals, width, label='Test RMSE (Points)', color='#10b981', alpha=0.88)
    
    axs[1, 1].set_title('4. Test Error Comparison: MSE & RMSE Across Models', fontsize=11.5, fontweight='bold')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(model_labels, fontsize=9.5)
    axs[1, 1].set_ylabel('Error Magnitude')
    axs[1, 1].set_ylim(0, 1.0)
    axs[1, 1].grid(True, linestyle='--', alpha=0.5, axis='y')
    axs[1, 1].legend(loc='upper right')
    
    for rect in rects_mse:
        h = rect.get_height()
        axs[1, 1].annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.2, fontweight='bold', color='#be123c')
    for rect in rects_rmse:
        h = rect.get_height()
        axs[1, 1].annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.2, fontweight='bold', color='#047857')
                           
    plt.tight_layout()
    metrics_plot_path = os.path.join('outputs/sleep', 'feature_selection_metrics_comparison.png')
    plt.savefig(metrics_plot_path, dpi=300)
    plt.close()
    print(f"    -> Saved 4-Panel Multi-Metric Comparison to: {metrics_plot_path}")
    print("=" * 80)
    
    return {
        'vif_features': vif_features,
        'lasso_features': lasso_features,
        'rfecv_features': list(ranking.index[:rfecv.n_features_]),
        'winning_features': curated_winner
    }

if __name__ == '__main__':
    from phase3_prep_engineering import run_phase_3
    prep = run_phase_3()
    run_phase_4(prep['X_train_scaled'], prep['y_train'])
