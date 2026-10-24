"""
CRISP-DM Phase 1: Business & Life Understanding
Sleep Health & Quality Multiple Linear Regression Project
"""

def run_phase_1():
    print("=" * 80)
    print("CRISP-DM PHASE 1: BUSINESS & SLEEP HEALTH UNDERSTANDING")
    print("=" * 80)
    print("\n1.1 Analytical Objectives:")
    print("  - Construct an interpretable, parsimonious Ordinary Least Squares (OLS) Multiple Linear")
    print("    Regression model to predict subjective and physiological Sleep Quality Score (1.0 to 10.0).")
    print("  - Optimize feature dimensionality down to an optimal, high-signal subset (8 to 12 features),")
    print("    eliminating multicollinearity while preserving maximum explanatory variance.")
    print("  - Provide transparent econometric coefficients for clinical sleep hygiene guidance.")
    
    print("\n1.2 Daily Life / Human Insights Objectives:")
    print("  - Quantify the relative longevity and sleep quality dividends of behavioral habits")
    print("    (caffeine, alcohol, evening screen exposure, nap duration) versus physiological/clinical factors")
    print("    (nocturnal awakenings, sleep latency, REM/deep sleep ratios, resting heart rate).")
    print("  - Evaluate the circadian penalty of irregular shift work and occupational stress.")
    print("  - Provide actionable guidelines for individuals aiming to optimize restorative sleep.")
    
    print("\n1.3 Success Criteria:")
    print("  - Explanatory Power: Cross-Validated / Test R^2 >= 0.70")
    print("  - Generalization Error: Test RMSE <= 0.80 points (on a 1.0 - 10.0 scale)")
    print("  - Multi-Metric Funnel: Simultaneous optimization of R^2, Adj R^2, MSE, and RMSE")
    print("  - Collinearity Elimination: Variance Inflation Factor (VIF) < 5.0 for all retained features")
    print("  - Statistical Significance: p < 0.05 for all final predictors")
    print("  - Econometric Validity: Satisfy 4 core OLS assumptions (Linearity, Normality, Homoscedasticity, Independence)")
    print("=" * 80)

if __name__ == "__main__":
    run_phase_1()
