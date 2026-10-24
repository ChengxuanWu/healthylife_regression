import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera

# Create outputs and models directory
os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Set style
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.family': 'sans-serif'})

print("--- CRISP-DM Phase 2 & 3: Data Preparation & Feature Engineering ---")
df = pd.read_csv('Life Expectancy Data.csv')
print(f"Loaded raw dataset shape: {df.shape}")

# Column cleaning
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_').str.replace('-', '_')

# Drop rows missing target
df = df.dropna(subset=['life_expectancy']).reset_index(drop=True)
print(f"Dataset after dropping missing target: {df.shape}")

# Median group imputation by status
continuous_cols = df.select_dtypes(include=[np.number]).columns
for col in continuous_cols:
    if col != 'life_expectancy':
        df[col] = df.groupby('status')[col].transform(lambda x: x.fillna(x.median()))
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

# Domain feature engineering
df['log_gdp'] = np.log1p(df['gdp'])
df['log_hiv'] = np.log1p(df['hiv_aids'])
df['human_capital'] = df['schooling'] * df['income_composition_of_resources']
df['immunization_index'] = (df['polio'] + df['diphtheria'] + df['hepatitis_b']) / 3.0

# ----------------- Candidate Feature Pools -----------------
# 1. Raw Baseline features (M1)
raw_features = [
    'adult_mortality', 'infant_deaths', 'alcohol', 'percentage_expenditure',
    'hepatitis_b', 'measles', 'bmi', 'under_five_deaths', 'polio',
    'total_expenditure', 'diphtheria', 'hiv_aids', 'gdp', 'population',
    'thinness__1_19_years', 'thinness_5_9_years',
    'income_composition_of_resources', 'schooling', 'status'
]
X_raw_df = pd.get_dummies(df[raw_features], columns=['status'], drop_first=True)
y = df['life_expectancy']

# 2. Curated Feature Pool (Engineered transformations resolving skew & raw collinearity)
# Drops built-in duplicates (infant_deaths, thinness_5_9, raw gdp, raw hiv_aids, raw individual vaccines)
curated_features = [
    'adult_mortality', 'under_five_deaths', 'alcohol', 'percentage_expenditure',
    'measles', 'bmi', 'total_expenditure', 'log_hiv', 'log_gdp', 'population',
    'thinness__1_19_years', 'human_capital', 'immunization_index', 'status'
]
X_cur_df = pd.get_dummies(df[curated_features], columns=['status'], drop_first=True)

# Train/Test Split (80/20, seed 42)
X_raw_train, X_raw_test, y_train, y_test = train_test_split(X_raw_df, y, test_size=0.20, random_state=42)
X_cur_train, X_cur_test, _, _ = train_test_split(X_cur_df, y, test_size=0.20, random_state=42)

# StandardScaler
scaler_raw = StandardScaler()
X_raw_train_s = pd.DataFrame(scaler_raw.fit_transform(X_raw_train), columns=X_raw_train.columns, index=X_raw_train.index)
X_raw_test_s = pd.DataFrame(scaler_raw.transform(X_raw_test), columns=X_raw_test.columns, index=X_raw_test.index)

scaler_cur = StandardScaler()
X_cur_train_s = pd.DataFrame(scaler_cur.fit_transform(X_cur_train), columns=X_cur_train.columns, index=X_cur_train.index)
X_cur_test_s = pd.DataFrame(scaler_cur.transform(X_cur_test), columns=X_cur_test.columns, index=X_cur_test.index)

# ----------------- Phase 4: Feature Selection -----------------
print("\n--- CRISP-DM Phase 4: Feature Selection Engineering ---")

def drop_high_vif(X, threshold=5.0):
    X_curr = X.copy()
    dropped = []
    while True:
        vifs = [variance_inflation_factor(X_curr.values, i) for i in range(X_curr.shape[1])]
        max_vif = max(vifs)
        if max_vif > threshold:
            drop_col = X_curr.columns[vifs.index(max_vif)]
            dropped.append((drop_col, float(max_vif)))
            print(f"Dropping '{drop_col}' (VIF: {max_vif:.2f})")
            X_curr = X_curr.drop(columns=[drop_col])
        else:
            break
    return X_curr, dropped

# Step 1: VIF Filter on Curated Pool
print("Step 1: Running Iterative VIF filter...")
X_train_vif, dropped_vif = drop_high_vif(X_cur_train_s, threshold=5.0)
vif_features = list(X_train_vif.columns)
print(f"VIF filter retained {len(vif_features)} features: {vif_features}")

# Step 2: LassoCV (Embedded)
print("\nStep 2: Running LassoCV on VIF-filtered features...")
lasso = LassoCV(cv=5, random_state=42).fit(X_train_vif, y_train)
lasso_features = list(X_train_vif.columns[lasso.coef_ != 0])
print(f"Optimal Alpha: {lasso.alpha_:.4f}")
print(f"LassoCV retained {len(lasso_features)} features: {lasso_features}")

# Step 3: RFECV (Wrapper)
print("\nStep 3: Running RFECV with LinearRegression...")
rfecv = RFECV(
    estimator=LinearRegression(),
    step=1,
    cv=5,
    scoring='neg_root_mean_squared_error'
)
rfecv.fit(X_train_vif, y_train)
rfecv_features = list(X_train_vif.columns[rfecv.support_])
print(f"RFECV Optimal count: {rfecv.n_features_}")
print(f"RFECV Selected features: {rfecv_features}")

# Model Candidates:
# M1: Baseline Raw OLS (19 raw features)
# M2: VIF Filter Only
# M3: VIF + LassoCV
# M4: VIF + RFECV / Parsimonious Model
models_dict = {
    'M1: Baseline Raw OLS': (X_raw_train_s, X_raw_test_s, list(X_raw_train_s.columns)),
    'M2: Filter Only (VIF)': (X_train_vif, X_cur_test_s[vif_features], vif_features),
    'M3: VIF + LassoCV': (X_cur_train_s[lasso_features], X_cur_test_s[lasso_features], lasso_features),
    'M4: VIF + RFECV': (X_cur_train_s[rfecv_features], X_cur_test_s[rfecv_features], rfecv_features)
}

results = []
trained_models = {}

for name, (X_tr, X_te, feats) in models_dict.items():
    X_tr_const = sm.add_constant(X_tr)
    X_te_const = sm.add_constant(X_te)
    ols = sm.OLS(y_train, X_tr_const).fit()
    trained_models[name] = ols
    
    y_pred = ols.predict(X_te_const)
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    p = len(feats)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    results.append({
        'Model Candidate': name,
        'Feature Count': p,
        'Test R2': round(float(r2), 4),
        'Test Adj R2': round(float(adj_r2), 4),
        'Test RMSE (Years)': round(float(rmse), 4),
        'Test MAE (Years)': round(float(mae), 4),
        'BIC': round(float(ols.bic), 1),
        'Decision': 'Selected Winner' if name == 'M4: VIF + RFECV' else ('High collinearity' if name.startswith('M1') else 'Evaluated Candidate')
    })

results_df = pd.DataFrame(results)
print("\n--- CRISP-DM Phase 6: Model Architecture Comparison Table ---")
print(results_df.to_string(index=False))

# Winning Model Diagnostics
winning_name = 'M4: VIF + RFECV'
winning_model = trained_models[winning_name]
winning_feats = rfecv_features
print("\n--- CRISP-DM Phase 5: Winning Model Statistical Summary ---")
print(winning_model.summary())

# Diagnostic tests
resid = winning_model.resid
fitted = winning_model.fittedvalues

# 1. Linearity
# Evaluated via Scatter of observed vs fitted
y_train_pred = winning_model.predict(sm.add_constant(X_cur_train_s[winning_feats]))

# 2. Normality of residuals (Jarque-Bera)
jb_stat, jb_pval, skew, kurt = jarque_bera(resid)
print(f"Jarque-Bera: stat={jb_stat:.2f}, p-value={jb_pval:.4e}, Skewness={skew:.3f}, Kurtosis={kurt:.3f}")

# 3. Homoscedasticity (Breusch-Pagan)
bp_test = het_breuschpagan(resid, sm.add_constant(X_cur_train_s[winning_feats]))
print(f"Breusch-Pagan LM Stat={bp_test[0]:.2f}, p-value={bp_test[1]:.4e}")

# 4. Independence of Errors (Durbin-Watson)
dw = durbin_watson(resid)
print(f"Durbin-Watson Statistic={dw:.4f}")

# Save diagnostic plots
fig, axs = plt.subplots(2, 2, figsize=(14, 11))

# Plot 1: Observed vs Predicted (Linearity)
axs[0, 0].scatter(y_train, y_train_pred, alpha=0.4, color='#1f77b4', edgecolors='none', s=25)
min_val = min(y_train.min(), y_train_pred.min())
max_val = max(y_train.max(), y_train_pred.max())
axs[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='45° Perfect Linearity')
axs[0, 0].set_title('1. Linearity Audit: Observed vs Predicted y', fontsize=12, fontweight='bold')
axs[0, 0].set_xlabel('Actual Life Expectancy (Years)')
axs[0, 0].set_ylabel('Predicted Life Expectancy (Years)')
axs[0, 0].legend()

# Plot 2: Q-Q Plot (Normality)
sm.qqplot(resid, line='45', fit=True, ax=axs[0, 1])
axs[0, 1].get_lines()[0].set_markerfacecolor('#2ca02c')
axs[0, 1].get_lines()[0].set_markeredgecolor('none')
axs[0, 1].get_lines()[0].set_alpha(0.5)
axs[0, 1].set_title(f'2. Normality Audit: Q-Q Plot (JB p={jb_pval:.2e})', fontsize=12, fontweight='bold')

# Plot 3: Residuals vs Fitted (Homoscedasticity)
axs[1, 0].scatter(fitted, resid, alpha=0.4, color='#ff7f0e', edgecolors='none', s=25)
axs[1, 0].axhline(0, color='red', linestyle='--', lw=2)
axs[1, 0].set_title('3. Homoscedasticity Audit: Residuals vs Fitted', fontsize=12, fontweight='bold')
axs[1, 0].set_xlabel('Fitted Values')
axs[1, 0].set_ylabel('Residuals (e)')

# Plot 4: Coefficient Weights (Domain Impact)
coefs = winning_model.params.drop('const')
coefs_sorted = coefs.sort_values()
colors = ['#d62728' if x < 0 else '#2ca02c' for x in coefs_sorted]
axs[1, 1].barh(coefs_sorted.index, coefs_sorted.values, color=colors, alpha=0.85)
axs[1, 1].axvline(0, color='black', lw=1)
axs[1, 1].set_title('4. Standardized Beta Coefficients (Effect Size)', fontsize=12, fontweight='bold')
axs[1, 1].set_xlabel('Standardized Beta (Years impact per 1 SD change)')

plt.tight_layout()
plot_path = os.path.join('outputs', 'regression_diagnostics.png')
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"Saved diagnostic plots to: {plot_path}")

# VIFs of winning model
vif_final = pd.Series([variance_inflation_factor(X_cur_train_s[winning_feats].values, i) for i in range(len(winning_feats))], index=winning_feats)
print("\nFinal Model VIFs:")
print(vif_final)

# Save artifacts for Phase 7 Deployment
scaler_for_prod = StandardScaler()
# We fit scaler on raw values of the winning features
scaler_for_prod.fit(X_cur_df[winning_feats])
joblib.dump(scaler_for_prod, 'models/scaler.pkl')
joblib.dump(winning_model, 'models/ols_model.pkl')
with open('models/feature_metadata.json', 'w') as f:
    json.dump({
        'selected_features': winning_feats,
        'coefficients': winning_model.params.to_dict(),
        'pvalues': winning_model.pvalues.to_dict(),
        'metrics': results_df.to_dict(orient='records')
    }, f, indent=4)

print("\nModel artifacts saved successfully in models/ directory!")
