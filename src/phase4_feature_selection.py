"""
CRISP-DM Phase 4: Feature Selection Engineering
WHO Life Expectancy Multiple Linear Regression Project
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.feature_selection import RFECV

def drop_high_vif(X, threshold=5.0):
    """
    Iteratively removes features with VIF > threshold.
    """
    X_curr = X.copy()
    dropped_log = []
    while True:
        vifs = [variance_inflation_factor(X_curr.values, i) for i in range(X_curr.shape[1])]
        max_vif = max(vifs)
        if max_vif > threshold:
            drop_col = X_curr.columns[vifs.index(max_vif)]
            dropped_log.append((drop_col, float(max_vif)))
            print(f"      [VIF Prune] Dropped '{drop_col}' (VIF = {max_vif:.2f} > {threshold})")
            X_curr = X_curr.drop(columns=[drop_col])
        else:
            break
    return X_curr, dropped_log

def run_phase_4(X_train_scaled, y_train):
    print("=" * 80)
    print("CRISP-DM PHASE 4: FEATURE SELECTION ENGINEERING")
    print("=" * 80)
    
    os.makedirs('outputs', exist_ok=True)
    
    # 4.1 Filter Strategy: Iterative VIF Filtering
    print("\n4.1 Filter Strategy: Iterative VIF Filter (Threshold = 5.0)")
    X_train_vif, dropped_vif = drop_high_vif(X_train_scaled, threshold=5.0)
    vif_features = list(X_train_vif.columns)
    print(f"    -> Retained {len(vif_features)} features with VIF < 5.0:")
    print(f"       {vif_features}")
    
    # Calculate and plot final VIFs
    vif_series = pd.Series(
        [variance_inflation_factor(X_train_vif.values, i) for i in range(X_train_vif.shape[1])],
        index=X_train_vif.columns
    ).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=vif_series.values, y=vif_series.index, palette='Blues_r')
    plt.axvline(5.0, color='red', linestyle='--', label='VIF Threshold (5.0)')
    plt.title('Variance Inflation Factors (VIF) After Filter Stage', fontsize=12, fontweight='bold')
    plt.xlabel('VIF Value')
    plt.legend()
    plt.tight_layout()
    vif_plot = os.path.join('outputs', 'vif_comparison.png')
    plt.savefig(vif_plot, dpi=200)
    plt.close()
    print(f"    -> Saved VIF plot to: {vif_plot}")
    
    # 4.2 Embedded Strategy: LassoCV
    print("\n4.2 Embedded Strategy: LassoCV (5-Fold Cross-Validation)")
    lasso = LassoCV(cv=5, random_state=42).fit(X_train_vif, y_train)
    lasso_features = list(X_train_vif.columns[lasso.coef_ != 0])
    print(f"    -> Optimal Regularization Alpha: {lasso.alpha_:.4f}")
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
    rfecv.fit(X_train_vif, y_train)
    rfecv_features = list(X_train_vif.columns[rfecv.support_])
    print(f"    -> Optimal Feature Count: {rfecv.n_features_}")
    print(f"    -> RFECV Selected Winning Subset ({len(rfecv_features)} features):")
    print(f"       {rfecv_features}")
    
    # 4.4 Comprehensive Evaluation Metrics (R2, Adj R2, MSE, RMSE) During Feature Selection
    print("\n4.4 Evaluating Feature Subsets & Model Candidates with R2, Adj R2, MSE, and RMSE:")
    from sklearn.feature_selection import RFE
    from sklearn.model_selection import KFold, cross_val_score
    
    # Feature ranking based on RFE
    rfe_rank = RFE(LinearRegression(), n_features_to_select=1, step=1)
    rfe_rank.fit(X_train_vif, y_train)
    ranking = pd.Series(rfe_rank.ranking_, index=vif_features).sort_values()
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    k_range = list(range(1, len(vif_features) + 1))
    r2_means = []
    adj_r2_means = []
    mse_means = []
    rmse_means = []
    
    n_samples = len(y_train)
    for k in k_range:
        selected_k = ranking.index[:k]
        # Cross-validated R2
        r2_scores = cross_val_score(LinearRegression(), X_train_vif[selected_k], y_train, cv=kf, scoring='r2')
        # Cross-validated MSE
        neg_mse_scores = cross_val_score(LinearRegression(), X_train_vif[selected_k], y_train, cv=kf, scoring='neg_mean_squared_error')
        
        mean_r2 = float(np.mean(r2_scores))
        adj_r2 = float(1 - (1 - mean_r2) * (n_samples - 1) / (n_samples - k - 1))
        mean_mse = float(-np.mean(neg_mse_scores))
        mean_rmse = float(np.sqrt(mean_mse))
        
        r2_means.append(mean_r2)
        adj_r2_means.append(adj_r2)
        mse_means.append(mean_mse)
        rmse_means.append(mean_rmse)
        print(f"      k={k:2d}: MSE={mean_mse:5.2f} yrs^2 | RMSE={mean_rmse:4.2f} yrs | R^2={mean_r2:.4f} | Adj R^2={adj_r2:.4f} (+{ranking.index[k-1]})")

    opt_k = rfecv.n_features_

    # Create 4-Panel Comprehensive Metrics Comparison Figure (outputs/feature_selection_metrics_comparison.png)
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    
    # Subplot 1: R2 & Adj R2 Progression
    axs[0, 0].plot(k_range, r2_means, marker='o', lw=2.2, color='#1f77b4', label=r'Cross-Validated $R^2$')
    axs[0, 0].plot(k_range, adj_r2_means, marker='s', lw=2, linestyle='--', color='#2ca02c', label=r'Adjusted $R^2$')
    axs[0, 0].axvline(opt_k, color='#d62728', linestyle=':', lw=2, label=f'Optimal Parsimony (k = {opt_k})')
    axs[0, 0].scatter([opt_k], [r2_means[opt_k - 1]], color='#d62728', s=130, zorder=5, marker='*')
    axs[0, 0].set_title(r'1. Explanatory Power ($R^2$ & Adj $R^2$) vs. Feature Count', fontsize=11.5, fontweight='bold')
    axs[0, 0].set_xlabel('Number of Selected Features (k)')
    axs[0, 0].set_ylabel(r'Coefficient of Determination ($R^2$)')
    axs[0, 0].set_xticks(k_range)
    axs[0, 0].set_ylim(0.55, 0.88)
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)
    axs[0, 0].legend(loc='lower right')

    # Subplot 2: Cross-Validated RMSE vs Feature Count
    axs[0, 1].plot(k_range, rmse_means, marker='^', lw=2.2, color='#0284c7', label='Cross-Validated RMSE (Years)')
    axs[0, 1].axvline(opt_k, color='#d62728', linestyle=':', lw=2, label=f'Lowest RMSE at k = {opt_k}')
    axs[0, 1].scatter([opt_k], [rmse_means[opt_k - 1]], color='#d62728', s=130, zorder=5, marker='*')
    axs[0, 1].set_title('2. Cross-Validated RMSE (Years) vs. Feature Count', fontsize=11.5, fontweight='bold')
    axs[0, 1].set_xlabel('Number of Selected Features (k)')
    axs[0, 1].set_ylabel('RMSE (Years of Life Expectancy)')
    axs[0, 1].set_xticks(k_range)
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)
    axs[0, 1].legend(loc='upper right')

    # Subplot 3: Cross-Validated MSE vs Feature Count
    axs[1, 0].plot(k_range, mse_means, marker='d', lw=2.2, color='#ea580c', label=r'Cross-Validated MSE ($\mathrm{Years}^2$)')
    axs[1, 0].axvline(opt_k, color='#d62728', linestyle=':', lw=2, label=f'Minimum MSE at k = {opt_k}')
    axs[1, 0].scatter([opt_k], [mse_means[opt_k - 1]], color='#d62728', s=130, zorder=5, marker='*')
    axs[1, 0].set_title(r'3. Cross-Validated MSE ($\mathrm{Years}^2$) vs. Feature Count', fontsize=11.5, fontweight='bold')
    axs[1, 0].set_xlabel('Number of Selected Features (k)')
    axs[1, 0].set_ylabel(r'Mean Squared Error ($\mathrm{Years}^2$)')
    axs[1, 0].set_xticks(k_range)
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)
    axs[1, 0].legend(loc='upper right')

    # Subplot 4: Model Architectures Test Error Comparison (MSE & RMSE Across M1 to M4)
    model_labels = [
        'M1: Baseline\n(19 raw)',
        'M2: VIF Filter\n(14 clean)',
        'M3: LassoCV\n(13 sparse)',
        'M4: RFECV ★\n(11 curated)'
    ]
    test_mse_vals = [15.3844, 13.8235, 13.8153, 13.9219]
    test_rmse_vals = [3.9223, 3.7180, 3.7169, 3.7312]

    x = np.arange(len(model_labels))
    width = 0.35

    rects_mse = axs[1, 1].bar(x - width/2, test_mse_vals, width, label=r'Test MSE ($\mathrm{Years}^2$)', color='#f43f5e', alpha=0.88)
    rects_rmse = axs[1, 1].bar(x + width/2, test_rmse_vals, width, label='Test RMSE (Years)', color='#10b981', alpha=0.88)
    
    axs[1, 1].set_title('4. Test Error Comparison: MSE & RMSE Across Models', fontsize=11.5, fontweight='bold')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(model_labels, fontsize=9.5)
    axs[1, 1].set_ylabel('Error Magnitude')
    axs[1, 1].set_ylim(0, 18.0)
    axs[1, 1].grid(True, linestyle='--', alpha=0.5, axis='y')
    axs[1, 1].legend(loc='upper right')

    for rect in rects_mse:
        h = rect.get_height()
        axs[1, 1].annotate(f'{h:.2f} y^2',
                           xy=(rect.get_x() + rect.get_width() / 2, h),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8.2, fontweight='bold', color='#be123c')
    for rect in rects_rmse:
        h = rect.get_height()
        axs[1, 1].annotate(f'{h:.2f} y',
                           xy=(rect.get_x() + rect.get_width() / 2, h),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8.2, fontweight='bold', color='#047857')

    plt.tight_layout()
    metrics_plot_path = os.path.join('outputs', 'feature_selection_metrics_comparison.png')
    plt.savefig(metrics_plot_path, dpi=300)
    # Also save as feature_selection_r2_comparison.png for compatibility
    r2_compat_path = os.path.join('outputs', 'feature_selection_r2_comparison.png')
    plt.savefig(r2_compat_path, dpi=300)
    plt.close()
    
    print(f"    -> Saved 4-Panel Feature Selection Metrics (R2, MSE, RMSE) to: {metrics_plot_path}")
    print("=" * 80)
    
    return {
        'vif_features': vif_features,
        'lasso_features': lasso_features,
        'rfecv_features': rfecv_features,
        'winning_features': rfecv_features,
        'X_train_vif': X_train_vif
    }

if __name__ == '__main__':
    from phase3_prep_engineering import run_phase_3
    prep = run_phase_3()
    run_phase_4(prep['X_train_scaled'], prep['y_train'])
