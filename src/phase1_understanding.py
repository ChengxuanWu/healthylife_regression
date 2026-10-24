"""
CRISP-DM Phase 1: Business & Life Understanding
WHO Life Expectancy Multiple Linear Regression Project
"""

def run_phase_1():
    print("=" * 80)
    print("CRISP-DM PHASE 1: BUSINESS & LIFE UNDERSTANDING")
    print("=" * 80)
    
    print("\n1.1 Analytical Objectives:")
    print("  - Target Variable: life_expectancy (Continuous, in years).")
    print("  - Model Class: Parsimonious, interpretable Multiple Linear Regression (OLS).")
    print("  - Dimension Goal: Condense 21 candidate features down to an optimal subset of 8-12 core predictors.")
    print("  - Explanatory Preservation: Retain statistical explanatory power without collinear inflation.")
    
    print("\n1.2 Daily Life / Human Insights Objectives:")
    print("  - Modifiable vs Macro Infrastructure: Quantify individual lifestyle factors (education, alcohol,")
    print("    nutrition/thinness, BMI) vs macro infrastructure (GDP, health expenditure, vaccination coverage).")
    print("  - Diminishing Returns: Identify economic thresholds where income gains taper off and lifestyle")
    print("    determinants take precedence.")
    print("  - Compound Human Capital: Measure the synergistic interaction of education and resource availability.")
    
    print("\n1.3 Formal Success Criteria:")
    print("  [x] Statistical Fit:       Adjusted R^2 >= 0.82")
    print("  [x] Prediction Accuracy:   Test RMSE <= 3.8 years")
    print("  [x] Multicollinearity:     VIF < 5.0 for all retained features")
    print("  [x] Significance:          p < 0.05 for all final regression coefficients")
    print("  [x] Econometric Rigor:     All 4 core OLS assumptions passed (Linearity, Normality,")
    print("                             Homoscedasticity, Independence of errors [1.5 <= d <= 2.5])")
    
    print("\n1.4 Mathematical OLS Model Formulation:")
    print("  y_i = beta_0 + sum(beta_j * X_ij) + epsilon_i,   where epsilon_i ~ i.i.d. N(0, sigma^2)")
    print("=" * 80)
    return True

if __name__ == '__main__':
    run_phase_1()
