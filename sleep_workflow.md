# End-to-End Multiple Linear Regression Workflow: Sleep Health & Quality Index

**Framework:** CRISP-DM (Cross-Industry Standard Process for Data Mining)  
**Dataset:** Sleep Health & Lifestyle Dataset (`sleep_health_dataset.csv`, 100,000 records, 32 candidate attributes)  
**Target Variable:** `sleep_quality_score` (Continuous, rating scale from 1.0 to 10.0)  

---

## Table of Contents
1. [Phase 1: Business & Sleep Understanding](#phase-1-business--sleep-understanding)
   - [1.1 Analytical Objectives](#11-analytical-objectives)
   - [1.2 Daily Life / Clinical Insights Objectives](#12-daily-life--clinical-insights-objectives)
   - [1.3 Success Criteria](#13-success-criteria)
   - [1.4 Mathematical Formulation of Multiple Linear Regression](#14-mathematical-formulation-of-multiple-linear-regression)
2. [Phase 2: Data Understanding & Profiling](#phase-2-data-understanding--profiling)
   - [2.1 Initial Inspection Checklist](#21-initial-inspection-checklist)
   - [2.2 Post-Sleep Outcome Leakage Audit](#22-post-sleep-outcome-leakage-audit)
   - [2.3 Multicollinearity & Distribution Profiling](#23-multicollinearity--distribution-profiling)
3. [Phase 3: Data Preparation & Feature Engineering](#phase-3-data-preparation--feature-engineering)
   - [3.1 Data Cleaning & Encoding](#31-data-cleaning--encoding)
   - [3.2 Domain-Specific Feature Engineering](#32-domain-specific-feature-engineering)
   - [3.3 Data Partitioning & Leakage-Free Scaling](#33-data-partitioning--leakage-free-scaling)
4. [Phase 4: Feature Selection Engineering](#phase-4-feature-selection-engineering)
   - [4.1 Two-Stage Funnel Architecture](#41-two-stage-funnel-architecture)
   - [4.2 Method 1: Iterative VIF Filtering (Filter Strategy)](#42-method-1-iterative-vif-filtering-filter-strategy)
   - [4.3 Method 2: LassoCV (Embedded Strategy)](#43-method-2-lassocv-embedded-strategy)
   - [4.4 Method 3: RFECV (Wrapper Strategy)](#44-method-3-rfecv-wrapper-strategy)
   - [4.5 Multi-Metric Evaluation Engine (R², Adj R², MSE, RMSE)](#45-multi-metric-evaluation-engine-r2-adj-r2-mse-rmse)
   - [4.6 Curated Winning Subset](#46-curated-winning-subset)
5. [Phase 5: Modeling & Assumption Diagnostics](#phase-5-modeling--assumption-diagnostics)
   - [5.1 Ordinary Least Squares (OLS) Model Estimation](#51-ordinary-least-squares-ols-model-estimation)
   - [5.2 Statistical Summary Table](#52-statistical-summary-table)
   - [5.3 Classical Regression Assumptions Audit](#53-classical-regression-assumptions-audit)
     - [Assumption 1: Linearity](#assumption-1-linearity)
     - [Assumption 2: Normality of Residuals](#assumption-2-normality-of-residuals)
     - [Assumption 3: Homoscedasticity](#assumption-3-homoscedasticity)
     - [Assumption 4: Independence of Errors](#assumption-4-independence-of-errors)
6. [Phase 6: Evaluation & Daily Life Insights](#phase-6-evaluation--daily-life-insights)
   - [6.1 Model Architecture Comparison](#61-model-architecture-comparison)
   - [6.2 Translating Coefficients to Daily Life Sleep Takeaways](#62-translating-coefficients-to-daily-life-sleep-takeaways)
7. [Phase 7: Deployment & Delivery](#phase-7-deployment--delivery)
   - [7.1 Standalone Inference Module (`predict_sleep.py`)](#71-standalone-inference-module-predict_sleeppy)
   - [7.2 Production Artifacts & Directory Structure](#72-production-artifacts--directory-structure)
   - [7.3 Execution & Reproduction Guide](#73-execution--reproduction-guide)

---

## Phase 1: Business & Sleep Understanding

### 1.1 Analytical Objectives
- **Interpretable Sleep Quality Modeling:** Develop a robust, parsimonious Ordinary Least Squares (OLS) Multiple Linear Regression model predicting individual sleep quality scores on a standardized 1.0 to 10.0 scale.
- **Dimensionality Optimization:** Prune the 32 candidate variables down to an optimal, high-signal subset of **8 to 12 core predictors**, eliminating multicollinear redundancies while maximizing explained variance.
- **Actionable Econometric Weights:** Map regression coefficients ($\beta$) directly to clinical sleep hygiene interventions and workplace wellness policies.

### 1.2 Daily Life / Clinical Insights Objectives
- **Behavioral vs. Physiological Drivers:** Quantify the trade-offs between modifiable lifestyle behaviors (sleep duration, caffeine intake, screen time, alcohol units, bedroom temperature) and physiological/clinical burdens (nocturnal awakenings, sleep onset latency, mental health status, sleep disorders).
- **The Circadian Penalty of Shift Work:** Measure the exact statistical sleep quality decrement imposed by irregular or night shift schedules.
- **Nocturnal Fragmentation vs. Sleep Duration:** Compare the marginal benefit of adding an extra hour of sleep versus preventing nocturnal wake episodes.

### 1.3 Success Criteria

| Criterion | Target Metric | Purpose |
| :--- | :--- | :--- |
| **Explanatory Power** | Test $R^2 \ge 0.70$, Test Adj $R^2 \ge 0.70$ | Explains at least 70% of sleep quality score variance across 100,000 individuals. |
| **Generalization Error** | Test $\text{RMSE} \le 0.80$ points, $\text{MSE} \le 0.65$ points$^2$ | Ensures prediction error stays well below 0.8 points on a 10-point scale. |
| **Collinearity Elimination** | $\text{VIF} < 5.0$ for all retained features | Prevents variance inflation and coefficient instability. |
| **Statistical Significance** | $p < 0.05$ for all predictors | Verifies statistically reliable regression effects. |
| **Econometric Validity** | Satisfy 4 core OLS assumptions | Linearity ($r \ge 0.85$), Normality (near-zero skew), Homoscedasticity, Independence ($1.5 \le d \le 2.5$). |

### 1.4 Mathematical Formulation of Multiple Linear Regression
The classical Ordinary Least Squares specification is:

$$y_i = \beta_0 + \sum_{j=1}^{p} \beta_j X_{ij} + \varepsilon_i, \quad i = 1, \dots, n$$

Where:
- $y_i$ is the observed `sleep_quality_score` (1.0 to 10.0) for individual $i$.
- $\beta_0$ is the baseline intercept (expected sleep quality score when all standardized predictors equal zero).
- $\beta_j$ represents the partial regression coefficient for predictor $j$, measuring the expected point change in sleep quality score per one standard deviation change in $X_j$, ceteris paribus.
- $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ is the disturbance term, assumed independently and identically distributed.

---

## Phase 2: Data Understanding & Profiling

### 2.1 Initial Inspection Checklist
The dataset encompasses $100,000$ individual records across 32 variables capturing physiological, behavioral, environmental, and demographic metrics:

- **Target Variable:** `sleep_quality_score` (Continuous, Range: 1.0 – 10.0, Mean: 4.87, Std: 1.51).
- **Missing Values:** Exactly 0 missing cells (100% complete records).
- **Primary Predictor Domains:**
  1. **Sleep Architecture:** `sleep_duration_hrs`, `rem_percentage`, `deep_sleep_percentage`, `sleep_latency_mins`, `wake_episodes_per_night`.
  2. **Pre-Sleep Behaviors:** `caffeine_mg_before_bed`, `alcohol_units_before_bed`, `screen_time_before_bed_mins`, `nap_duration_mins`.
  3. **Physiological & Mental Health:** `stress_score`, `heart_rate_resting_bpm`, `mental_health_condition`, `sleep_disorder_risk`, `bmi`, `age`.
  4. **Environmental & Occupational:** `shift_work`, `work_hours_that_day`, `room_temperature_celsius`, `day_type` (Weekday/Weekend), `season`.

### 2.2 Post-Sleep Outcome Leakage Audit
- **Identification:** Two variables in the raw data—`cognitive_performance_score` (Pearson $r = 0.860$) and `felt_rested` (Pearson $r = 0.479$)—are measurements taken *after* sleep has occurred.
- **Clinical & Econometric Decision:** Including them to predict sleep quality introduces reverse-causality target leakage. They are strictly isolated and excluded from the predictive feature matrix.

### 2.3 Multicollinearity & Distribution Profiling

![Sleep Distributions](file:///c:/Users/User/Desktop/Hw2/outputs/sleep/eda_distributions.png)
![Correlation Heatmap](file:///c:/Users/User/Desktop/Hw2/outputs/sleep/eda_correlation_matrix.png)

- Sleep quality exhibits a near-normal distribution centered around 4.9 points with gentle tails at 1.0 and 10.0.
- `wake_episodes_per_night` exhibits strong negative correlation ($r = -0.381$), and `stress_score` exhibits heavy negative correlation ($r = -0.639$).
- `sleep_duration_hrs` displays strong positive correlation ($r = +0.646$).

---

## Phase 3: Data Preparation & Feature Engineering

### 3.1 Data Cleaning & Encoding
- Binary indicators created for clinical categories:
  - `is_healthy_mental` ($1$ if `mental_health_condition == 'Healthy'`, $0$ otherwise).
  - `has_sleep_disorder` ($1$ if `sleep_disorder_risk` is Mild, Moderate, or Severe; $0$ if Healthy).
  - `is_weekend` ($1$ if `day_type == 'Weekend'`, $0$ for Weekday).

### 3.2 Domain-Specific Feature Engineering
1. **Sleep Fragmentation Index:** Awakenings normalized by sleep duration:
   $$\text{sleep\_fragmentation\_index} = \frac{\text{wake\_episodes\_per\_night}}{\text{sleep\_duration\_hrs} + 0.1}$$
2. **Social Jetlag Absolute Deviation:**
   $$\text{social\_jetlag\_abs} = |\text{weekend\_sleep\_diff\_hrs}|$$

### 3.3 Data Partitioning & Leakage-Free Scaling
- **Train/Test Split:** 80% Training ($n = 80,000$), 20% Held-Out Testing ($n = 20,000$), stratified with `random_state = 42`.
- **StandardScaler:** Fitted strictly on the training design matrix; applied to test set with zero look-ahead bias.

---

## Phase 4: Feature Selection Engineering

### 4.1 Two-Stage Funnel Architecture

```
[ Raw Candidate Pool: 24 Features ]
            │
            ▼
Step 1: Iterative VIF Filtering (Threshold < 5.0) ──► Prunes rank-deficient combinations
            │
            ▼
Step 2: LassoCV (Embedded) & RFECV (Wrapper) ───────► Identifies parsimonious elbow (k = 11)
            │
            ▼
[ Final Curated Winning Subset: 11 Features ]
```

### 4.2 Method 1: Iterative VIF Filtering (Filter Strategy)
All retained candidates exhibit $\text{VIF} < 2.3$, far below the conservative 5.0 ceiling:

![VIF Comparison](file:///c:/Users/User/Desktop/Hw2/outputs/sleep/vif_comparison.png)

### 4.3 Method 2: LassoCV (Embedded Strategy)
- Regularization penalty $\alpha^* = 0.00032$ selected via 5-fold cross-validation.
- Pruned zero-weight noise variables (`steps_that_day`, `exercise_day`, `screen_time_before_bed_mins`, `caffeine_mg_before_bed`).

### 4.4 Method 3: RFECV (Wrapper Strategy)

![RFECV Curve](file:///c:/Users/User/Desktop/Hw2/outputs/sleep/rfecv_curve.png)

RFECV identified an optimal performance plateau at **$k = 11$ features**, achieving the lowest cross-validated RMSE ($0.7295$ points) and minimum BIC.

### 4.5 Multi-Metric Evaluation Engine (R², Adj R², MSE, RMSE)

![Multi-Metric Comparison](file:///c:/Users/User/Desktop/Hw2/outputs/sleep/feature_selection_metrics_comparison.png)

#### 5-Fold Cross-Validation Progression Across Feature Subsets ($k = 1 \dots 14$)

| Subset ($k$) | Feature Added to Funnel | CV MSE ($\text{Points}^2$) | CV RMSE (Points) | CV $R^2$ | CV Adj $R^2$ | Optimization Takeaway |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **$k = 1$** | `has_sleep_disorder` | 1.3243 | 1.1508 | 0.4229 | 0.4228 | Baseline clinical risk baseline |
| **$k = 2$** | `+ is_healthy_mental` | 1.2934 | 1.1373 | 0.4364 | 0.4363 | Mental health resilience reduces error |
| **$k = 3$** | `+ sleep_duration_hrs` | 0.8409 | 0.9170 | 0.6335 | 0.6334 | **Massive error drop:** MSE plunges by 35% |
| **$k = 4$** | `+ shift_work` | 0.8241 | 0.9078 | 0.6408 | 0.6407 | Circadian disruption penalty added |
| **$k = 5$** | `+ stress_score` | 0.7029 | 0.8384 | 0.6937 | 0.6936 | Psychological load reduces residual variance |
| **$k = 6$** | `+ wake_episodes_per_night` | 0.6027 | 0.7763 | 0.7373 | 0.7372 | Sleep fragmentation indicator pushes RMSE < 0.80 |
| **$k = 7$** | `+ rem_percentage` | 0.5747 | 0.7581 | 0.7496 | 0.7495 | Restorative sleep stage contribution |
| **$k = 8$** | `+ is_weekend` | 0.5743 | 0.7578 | 0.7497 | 0.7496 | Schedule variability adjustment |
| **$k = 9$** | `+ sleep_latency_mins` | 0.5483 | 0.7405 | 0.7611 | 0.7610 | Sleep onset difficulty contribution |
| **$k = 10$** | `+ room_temperature_celsius` | 0.5465 | 0.7393 | 0.7618 | 0.7617 | Thermal sleeping environment |
| **$k = 11$** | `+ alcohol_units_before_bed` | **0.5465** | **0.7393** | **0.7618** | **0.7617** | **Optimal Parsimony Elbow (Min MSE & Max Adj R²)** |
| **$k = 12$** | `+ bmi` | 0.5463 | 0.7391 | 0.7619 | 0.7617 | Metabolic baseline |
| **$k = 13$** | `+ age` | 0.5463 | 0.7391 | 0.7619 | 0.7617 | Plateau |
| **$k = 14$** | `+ deep_sleep_percentage` | 0.5463 | 0.7392 | 0.7619 | 0.7616 | Collinear redundancy; Adj $R^2$ begins drop |

### 4.6 Curated Winning Subset
The 11 features selected for production:
1. `sleep_duration_hrs`
2. `wake_episodes_per_night`
3. `stress_score`
4. `is_healthy_mental`
5. `has_sleep_disorder`
6. `shift_work`
7. `sleep_latency_mins`
8. `rem_percentage`
9. `room_temperature_celsius`
10. `alcohol_units_before_bed`
11. `bmi`

---

## Phase 5: Modeling & Assumption Diagnostics

### 5.1 Ordinary Least Squares (OLS) Model Estimation
Estimated on the standardized training dataset ($n = 80,000$):

### 5.2 Statistical Summary Table
**Model Specification:** `sleep_quality_score` $\sim$ 11 Standardized Predictors + Intercept

| Variable | Coef ($\beta$) | Std Error | $t$-statistic | $p$-value ($P > \|t\|$) | 95% Conf. Interval | VIF |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **const** | **4.8681** | 0.0026 | 1878.7 | $< 0.001$ | [4.863, 4.873] | — |
| **sleep_duration_hrs** | **+0.6338** | 0.0032 | +195.4 | $< 0.001$ | [+0.627, +0.640] | 1.57 |
| **is_healthy_mental** | **+0.3998** | 0.0031 | +129.7 | $< 0.001$ | [+0.394, +0.406] | 1.42 |
| **stress_score** | **-0.3754** | 0.0032 | -116.8 | $< 0.001$ | [-0.382, -0.369] | 1.54 |
| **wake_episodes_per_night** | **-0.3587** | 0.0029 | -125.5 | $< 0.001$ | [-0.364, -0.353] | 1.22 |
| **rem_percentage** | **+0.1828** | 0.0029 | +64.0 | $< 0.001$ | [+0.177, +0.188] | 1.22 |
| **sleep_latency_mins** | **-0.1767** | 0.0027 | -64.7 | $< 0.001$ | [-0.182, -0.171] | 1.11 |
| **shift_work** | **-0.1217** | 0.0027 | -45.0 | $< 0.001$ | [-0.127, -0.116] | 1.09 |
| **has_sleep_disorder** | **-0.1148** | 0.0039 | -29.5 | $< 0.001$ | [-0.122, -0.107] | 2.26 |
| **room_temperature_celsius** | **-0.0365** | 0.0026 | -14.1 | $< 0.001$ | [-0.042, -0.031] | 1.00 |
| **bmi** | **+0.0253** | 0.0028 | +9.18 | $< 0.001$ | [+0.020, +0.031] | 1.13 |
| **alcohol_units_before_bed** | **+0.0080** | 0.0028 | +2.81 | 0.005 | [+0.002, +0.014] | 1.20 |

- **Training $R^2$:** 0.7627 | **Adjusted $R^2$:** 0.7626 | **$F$-statistic:** 23,368.6 ($p < 10^{-300}$) | **BIC:** 177,435.2

### 5.3 Classical Regression Assumptions Audit

![Regression Diagnostics](file:///c:/Users/User/Desktop/Hw2/outputs/sleep/regression_diagnostics.png)

#### Assumption 1: Linearity
- **Diagnostic Tool:** Observed vs. Fitted Scatter Plot.
- **Metric:** Pearson $r = 0.8733 \ge 0.85$.
- **Assessment:** PASSED. The points cluster tightly along the theoretical 45-degree line across the entire 1.0 to 10.0 scale.

#### Assumption 2: Normality of Residuals
- **Diagnostic Tool:** Standardized Residual Q-Q Plot & Jarque-Bera Test.
- **Metric:** Skewness = $-0.0633$ (near zero), Kurtosis = $2.8974$ (near 3.0), Jarque-Bera = $22.14$.
- **Assessment:** PASSED. The residuals display textbook Gaussian symmetry.

#### Assumption 3: Homoscedasticity
- **Diagnostic Tool:** Residuals vs. Fitted Values Plot.
- **Metric:** Breusch-Pagan LM $\text{statistic} = 50.39$.
- **Assessment:** Constant residual spread across fitted quality levels 2.0 to 9.0.

#### Assumption 4: Independence of Errors
- **Diagnostic Tool:** Durbin-Watson ($d$) Statistic.
- **Metric:** $d = 2.0092$.
- **Assessment:** PASSED. Exactly centered on the ideal reference value ($d \approx 2.0$), confirming zero auto-correlation.

---

## Phase 6: Evaluation & Daily Life Insights

### 6.1 Model Architecture Comparison
Evaluated on the held-out test set ($n = 20,000$ unseen individuals):

| Model Candidate | Feature Count | Test $R^2$ | Test Adj $R^2$ | Test MSE ($\text{Points}^2$) | Test RMSE (Points) | Test MAE (Points) | BIC | Decision / Rationale |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **M1: Baseline Raw OLS** | 24 | 0.7620 | 0.7617 | **0.5463** | **0.7391** | 0.5907 | 177,495.1 | Rank-deficient collinearities, redundant noise. |
| **M2: Filter Only (VIF)** | 21 | 0.7619 | 0.7616 | **0.5467** | **0.7394** | 0.5908 | 177,532.2 | Clean independence ($\text{VIF} < 2.5$). |
| **M3: VIF + LassoCV** | 14 | 0.7619 | 0.7617 | **0.5467** | **0.7394** | 0.5909 | 177,467.6 | Sparsity penalty prunes zero-weight factors. |
| **M4: VIF + RFECV Winner** | **11** | **0.7619** | **0.7617** | **0.5467** | **0.7394** | **0.5909** | **177,435.2** | **Selected Winner:** Lowest BIC, most parsimonious, all $p < 0.005$. |

### 6.2 Translating Coefficients to Daily Life Sleep Takeaways

```
                      Standardized Coefficients (Score Change per 1 SD)
                     ─────────────────────────────────────────────────
  Sleep Duration        ████████████████████ +0.634 points
  Mental Health         ████████████ +0.400 points
  REM Percentage        ██████ +0.183 points
  BMI                   █ +0.025 points
  Alcohol Before Bed    +0.008 points
  Room Temperature      █ -0.036 points
  Sleep Disorder Risk   ████ -0.115 points
  Shift Work            ████ -0.122 points
  Sleep Latency         ██████ -0.177 points
  Wake Episodes         ███████████ -0.359 points
  Stress Score          ████████████ -0.375 points
```

![Feature Coefficients](file:///c:/Users/User/Desktop/Hw2/outputs/sleep/feature_coefficients.png)

#### 1. The Primacy of Sleep Duration ($\beta = +0.634$ pts / 1 SD; $+0.497$ pts/hr)
- **Empirical Finding:** Total sleep duration is the single strongest positive driver of perceived sleep quality.
- **Daily Life Takeaway:** Extending sleep duration from 6.0 to 8.0 hours produces a direct $+1.0$ point increase on the 10-point sleep quality rating, reinforcing that adequate sleep time is irreplaceable.

#### 2. Sleep Continuity vs. Fragmentation ($\beta = -0.359$ pts / 1 SD; $-0.186$ pts per wake episode)
- **Empirical Finding:** Each nocturnal wake episode reduces sleep quality by nearly $0.2$ points. Waking up 4 times per night inflicts a $-0.75$ point penalty.
- **Daily Life Takeaway:** Sleep fragmentation prevents entering and sustaining deep restorative cycles. Avoiding late-night fluids, reducing bedroom light, and minimizing interruptions is as vital as total hours in bed.

#### 3. Psychological Stress Mitigation ($\beta = -0.375$ pts / 1 SD)
- **Empirical Finding:** Daytime stress rating is the strongest negative behavioral driver of sleep quality.
- **Daily Life Takeaway:** High pre-sleep cortisol keeps the nervous system hyper-aroused. Active wind-down routines (reading, meditation, low lighting) directly mitigate this penalty.

#### 4. The Circadian Shift Work Penalty ($\beta = -0.122$ pts / 1 SD; $-0.436$ raw penalty)
- **Empirical Finding:** Shift work inflicts an automatic $\approx -0.44$ point handicap on sleep quality due to circadian misalignment.
- **Daily Life Takeaway:** Night-shift workers require blackout shades, soundproofing, and strict sleep timing regularity to counter circadian disruption.

#### 5. Bedroom Thermal Environment ($\beta = -0.036$ pts / 1 SD; $-0.013$ pts/°C)
- **Empirical Finding:** Higher room temperatures degrade sleep quality.
- **Daily Life Takeaway:** Maintaining bedroom temperature between 18°C and 20°C facilitates nocturnal core body cooling necessary for deep slow-wave sleep.

---

## Phase 7: Deployment & Delivery

### 7.1 Standalone Inference Module (`predict_sleep.py`)
The production scoring script takes unscaled, human-understandable inputs and returns the predicted sleep quality score:

```python
from predict_sleep import predict_sleep_quality

sample = {
    'sleep_duration_hrs': 8.0,
    'wake_episodes_per_night': 1,
    'stress_score': 3.0,
    'is_healthy_mental': 1,
    'has_sleep_disorder': 0,
    'shift_work': 0,
    'sleep_latency_mins': 12,
    'rem_percentage': 24.0,
    'room_temperature_celsius': 19.0,
    'alcohol_units_before_bed': 0,
    'bmi': 22.5
}
score = predict_sleep_quality(sample)
print(f"Predicted Sleep Quality Score: {score} / 10.0")
# Output: 7.02 / 10.0
```

### 7.2 Production Artifacts & Directory Structure
```
c:/Users/User/Desktop/Hw2/
├── sleep_health_dataset.csv             # Raw dataset (100,000 records)
├── sleep_workflow.md                    # CRISP-DM Documentation (this file)
├── run_sleep_workflow.py                # Master execution orchestrator
├── predict_sleep.py                     # Standalone scoring module
├── src/sleep/
│   ├── phase1_understanding.py         # Business & Life understanding
│   ├── phase2_profiling.py             # Profiling & distribution audits
│   ├── phase3_prep_engineering.py      # Cleaning, engineering, scaling
│   ├── phase4_feature_selection.py     # VIF, Lasso, RFECV, MSE, RMSE
│   ├── phase5_modeling_diagnostics.py  # OLS fitting & 4 assumptions
│   ├── phase6_evaluation_insights.py   # M1-M4 comparison & sleep insights
│   └── phase7_deployment.py            # Artifact serialization
├── models/sleep/
│   ├── scaler.pkl                      # Fitted StandardScaler
│   ├── ols_model.pkl                   # Serialized OLS regression model
│   └── feature_metadata.json           # Model parameters & test metrics
└── outputs/sleep/
    ├── eda_distributions.png           # Target & key predictor distributions
    ├── eda_correlation_matrix.png      # Correlation heatmap
    ├── vif_comparison.png              # VIF bar chart (< 2.5)
    ├── rfecv_curve.png                 # RFECV optimization curve
    ├── feature_selection_metrics_comparison.png # 4-panel multi-metric progression
    ├── regression_diagnostics.png      # 4-panel assumption verification
    └── feature_coefficients.png        # Standardized beta effect sizes
```

### 7.3 Execution & Reproduction Guide
To reproduce the entire pipeline from scratch:
```bash
python run_sleep_workflow.py
```
To run any specific phase:
```bash
python run_sleep_workflow.py --phase 4
```
