"""
Master Orchestrator: End-to-End CRISP-DM Pipeline for Sleep Health Dataset
"""

import sys
import argparse
import os

# Ensure local imports work cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.sleep.phase1_understanding import run_phase_1
from src.sleep.phase2_profiling import run_phase_2
from src.sleep.phase3_prep_engineering import run_phase_3
from src.sleep.phase4_feature_selection import run_phase_4
from src.sleep.phase5_modeling_diagnostics import run_phase_5
from src.sleep.phase6_evaluation_insights import run_phase_6
from src.sleep.phase7_deployment import run_phase_7

def main():
    parser = argparse.ArgumentParser(description="Run Sleep Health CRISP-DM Multiple Linear Regression Pipeline")
    parser.add_argument("--phase", type=int, choices=range(1, 8), help="Run a specific phase (1-7). Omit to run full pipeline.")
    args = parser.parse_args()
    
    print("\n" + "#" * 80)
    print("  SLEEP HEALTH & QUALITY INTELLIGENCE: CRISP-DM REGRESSION PIPELINE")
    print("  Dataset: sleep_health_dataset.csv (100,000 observations, 32 attributes)")
    print("  Target Variable: sleep_quality_score (Continuous, 1.0 to 10.0)")
    print("#" * 80 + "\n")
    
    if args.phase == 1:
        run_phase_1()
    elif args.phase == 2:
        run_phase_2()
    elif args.phase == 3:
        run_phase_3()
    elif args.phase == 4:
        prep = run_phase_3()
        run_phase_4(prep['X_train_scaled'], prep['y_train'])
    elif args.phase == 5:
        prep = run_phase_3()
        fs = run_phase_4(prep['X_train_scaled'], prep['y_train'])
        run_phase_5(prep['X_train_scaled'], prep['y_train'], fs['winning_features'])
    elif args.phase == 6:
        prep = run_phase_3()
        fs = run_phase_4(prep['X_train_scaled'], prep['y_train'])
        run_phase_6(
            prep['X_train_scaled'], prep['X_test_scaled'],
            prep['y_train'], prep['y_test'],
            fs['vif_features'], fs['lasso_features'], fs['winning_features']
        )
    elif args.phase == 7:
        prep = run_phase_3()
        fs = run_phase_4(prep['X_train_scaled'], prep['y_train'])
        eval_res = run_phase_6(
            prep['X_train_scaled'], prep['X_test_scaled'],
            prep['y_train'], prep['y_test'],
            fs['vif_features'], fs['lasso_features'], fs['winning_features']
        )
        run_phase_7(eval_res['winning_model'], fs['winning_features'], prep['df'], eval_res['eval_df'])
    else:
        # Full Pipeline Execution
        run_phase_1()
        df = run_phase_2()
        prep = run_phase_3()
        fs = run_phase_4(prep['X_train_scaled'], prep['y_train'])
        ols_model = run_phase_5(prep['X_train_scaled'], prep['y_train'], fs['winning_features'])
        eval_res = run_phase_6(
            prep['X_train_scaled'], prep['X_test_scaled'],
            prep['y_train'], prep['y_test'],
            fs['vif_features'], fs['lasso_features'], fs['winning_features']
        )
        metadata = run_phase_7(eval_res['winning_model'], fs['winning_features'], prep['df'], eval_res['eval_df'])
        
        print("\n" + "*" * 80)
        print("  SLEEP HEALTH PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print(f"  Final Model Test R^2: {eval_res['eval_df'].iloc[-1]['Test R2']}")
        print(f"  Final Model Test MSE: {eval_res['eval_df'].iloc[-1]['Test MSE (Points^2)']} pts^2")
        print(f"  Final Model Test RMSE: {eval_res['eval_df'].iloc[-1]['Test RMSE (Points)']} pts")
        print(f"  Production Artifacts Saved to: models/sleep/")
        print(f"  Diagnostic Charts Saved to: outputs/sleep/")
        print("*" * 80 + "\n")

if __name__ == '__main__':
    main()
