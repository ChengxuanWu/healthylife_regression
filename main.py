"""
WHO Life Expectancy: End-to-End CRISP-DM Workflow Execution Master Script
"""

import sys
import argparse
import os

# Ensure src is in python path
sys.path.insert(0, os.path.abspath('src'))

from phase1_understanding import run_phase_1
from phase2_profiling import run_phase_2
from phase3_prep_engineering import run_phase_3
from phase4_feature_selection import run_phase_4
from phase5_modeling_diagnostics import run_phase_5
from phase6_evaluation_insights import run_phase_6
from phase7_deployment import run_phase_7

def main():
    parser = argparse.ArgumentParser(description="WHO Life Expectancy CRISP-DM Pipeline")
    parser.add_argument('--phase', type=int, choices=range(1, 8), default=None,
                        help="Run a specific phase (1-7). Default is to run all phases.")
    args = parser.parse_args()

    print("\n" + "#" * 80)
    print("  WHO LIFE EXPECTANCY MULTIPLE LINEAR REGRESSION")
    print("  CRISP-DM FRAMEWORK END-TO-END EXECUTION")
    print("#" * 80 + "\n")

    if args.phase == 1 or args.phase is None:
        run_phase_1()
        
    if args.phase == 2 or args.phase is None:
        run_phase_2()
        
    if args.phase in [3, 4, 5, 6, 7] or args.phase is None:
        prep = run_phase_3()
        
    if args.phase in [4, 5, 6, 7] or args.phase is None:
        fs = run_phase_4(prep['X_train_scaled'], prep['y_train'])
        
    if args.phase in [5, 6, 7] or args.phase is None:
        m5 = run_phase_5(prep['X_train_scaled'], prep['y_train'], fs['winning_features'])
        
    if args.phase in [6, 7] or args.phase is None:
        e6 = run_phase_6(
            df=prep['df'],
            X_train_scaled=prep['X_train_scaled'],
            X_test_scaled=prep['X_test_scaled'],
            y_train=prep['y_train'],
            y_test=prep['y_test'],
            vif_feats=fs['vif_features'],
            lasso_feats=fs['lasso_features'],
            rfecv_feats=fs['rfecv_features']
        )
        
    if args.phase == 7 or args.phase is None:
        run_phase_7(
            X_cur=prep['X_cur'],
            winning_features=fs['winning_features'],
            winning_model=m5['model'],
            eval_df=e6['eval_df']
        )

    print("\n" + "#" * 80)
    print("  ALL REQUESTED CRISP-DM PHASES SUCCESSFULLY COMPLETED!")
    print("  Artifacts generated in outputs/ and models/.")
    print("#" * 80 + "\n")

if __name__ == '__main__':
    main()
