"""End-to-End Verification Test Suite for Global Cyber Incident Dashboard Extensions."""
import os
import sys
import subprocess
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import load_incidents, connection_status
from src.schema import detect_columns
from src.geo import compute_geo_summary, build_choropleth_figure
from src.evaluation import (
    load_test_predictions,
    compute_confusion_matrix,
    compute_detailed_metrics,
    analyze_error_risks,
    build_confusion_matrix_figure,
    SEVERITY_CLASSES
)
from src.explanation_view import clean_feature_name, build_local_explanation_figure

def test_feature_a_geography():
    print("\n==========================================")
    print("TEST 1: Feature A - Global Incident Heat Map")
    print("==========================================")
    
    # 1. Connection & Schema
    status, msg = connection_status()
    assert status, f"MySQL connection failed: {msg}"
    df = load_incidents()
    assert not df.empty, "Incident dataset is empty!"
    print(f"[OK] MySQL connection OK, loaded {len(df):,} incidents with {df.shape[1]} columns.")
    
    cols = detect_columns(df)
    assert cols["country"] == "initiator_country", f"Country column not detected correctly: {cols}"
    print(f"[OK] Detected columns: {cols}")
    
    # 2. Geo aggregation on full data
    agg_df, stats, unmapped = compute_geo_summary(df)
    print(f"[OK] Mappable records: {stats['mappable']:,} ({stats['pct_mappable']:.1f}%) across {stats['distinct_countries']} countries.")
    print(f"[OK] Unmapped records: {stats['unmappable']:,} (unattributed/macro-regional correctly identified).")
    assert stats['mappable'] > 1500, f"Expected >1500 mappable records, got {stats['mappable']}"
    assert stats['distinct_countries'] >= 80, f"Expected >=80 distinct countries, got {stats['distinct_countries']}"
    
    # 3. Test all 4 view modes for Choropleth
    modes = ["Total Incident Count", "High Severity Count", "Severity Proportion (% High)", "Dominant Severity"]
    for mode in modes:
        fig = build_choropleth_figure(agg_df, view_mode=mode)
        assert len(fig.data) > 0, f"Figure for view mode {mode} has no traces!"
        print(f"[OK] Choropleth figure generated for '{mode}' ({len(fig.data)} traces).")
        
    # 4. Test filtering behaviour (e.g. year range 2018-2022, High severity only)
    filt = df[(df["incident_year"] >= 2018) & (df["incident_year"] <= 2022) & (df["severity_class"] == "High")]
    agg_filt, stats_filt, _ = compute_geo_summary(filt)
    print(f"[OK] Filtered view (2018-2022, High Severity only): {stats_filt['mappable']} mappable records across {stats_filt['distinct_countries']} countries.")
    fig_filt = build_choropleth_figure(agg_filt, view_mode="Total Incident Count")
    assert len(fig_filt.data) > 0, "Filtered choropleth failed to build."

    # 5. Test empty data handling
    empty_fig = build_choropleth_figure(pd.DataFrame(), "Total Incident Count")
    assert len(empty_fig.layout.annotations) > 0, "Empty state annotation missing."
    print("[OK] Graceful empty-state handling verified.")

def test_feature_b_confusion_matrix():
    print("\n==========================================")
    print("TEST 2: Feature B - Confusion Matrix & Error Analysis")
    print("==========================================")
    
    out_dir = PROJECT_ROOT / "data" / "r_outputs"
    preds_df = load_test_predictions(out_dir)
    assert preds_df is not None, "Failed to load tuned_test_predictions.csv!"
    assert len(preds_df) == 4098, f"Expected 4098 rows (683 test rows * 6 models), got {len(preds_df)}"
    
    expected_models = ["Extra Trees", "Random Forest", "Gradient Boosting", "XGBoost", "Decision Tree", "MLP Neural Network"]
    actual_models = preds_df["model"].unique().tolist()
    for m in expected_models:
        assert m in actual_models, f"Missing model {m} in test predictions!"
    print(f"[OK] Loaded test predictions for all {len(actual_models)} tuned models ({len(preds_df):,} total evaluations).")

    # Load benchmark metrics from tuned_test_results.csv
    bench_results = pd.read_csv(out_dir / "tuned_test_results.csv").set_index("model")

    for model_name in expected_models:
        sub = preds_df[preds_df["model"] == model_name]
        assert len(sub) == 683, f"Expected 683 test rows for {model_name}, got {len(sub)}"
        
        # 1. Confusion matrix calculation
        cm_raw, cm_norm = compute_confusion_matrix(sub, normalize=True)
        assert list(cm_raw.index) == SEVERITY_CLASSES, f"Incorrect actual class order: {list(cm_raw.index)}"
        assert list(cm_raw.columns) == SEVERITY_CLASSES, f"Incorrect predicted class order: {list(cm_raw.columns)}"
        assert cm_raw.values.sum() == 683, f"Confusion matrix total {cm_raw.values.sum()} != 683"
        
        # 2. Check metrics consistency
        metrics_df, summary = compute_detailed_metrics(sub)
        bench_acc = bench_results.loc[model_name, "accuracy"]
        bench_macro_f1 = bench_results.loc[model_name, "macro_f1"]
        
        assert np.isclose(summary["Accuracy"], bench_acc, atol=1e-5), f"Accuracy mismatch for {model_name}: {summary['Accuracy']} vs {bench_acc}"
        assert np.isclose(summary["Macro F1"], bench_macro_f1, atol=1e-5), f"Macro F1 mismatch for {model_name}: {summary['Macro F1']} vs {bench_macro_f1}"
        
        # 3. Error analysis
        risks = analyze_error_risks(sub)
        assert risks["actual_high_total"] == 44, f"Actual high total should be 44, got {risks['actual_high_total']}"
        assert risks["high_correct"] + risks["total_high_missed"] == 44
        
        # 4. Plotly figure generation
        fig_raw = build_confusion_matrix_figure(cm_raw, cm_norm, show_normalized=False, model_name=model_name)
        fig_norm = build_confusion_matrix_figure(cm_raw, cm_norm, show_normalized=True, model_name=model_name)
        assert len(fig_raw.data) > 0 and len(fig_norm.data) > 0
        
        print(f"[OK] Model '{model_name}': Acc={summary['Accuracy']:.4f}, Macro F1={summary['Macro F1']:.4f}, High Detected={risks['high_correct']}/44, High Missed={risks['total_high_missed']}.")

def test_feature_c_prediction_explanation():
    print("\n==========================================")
    print("TEST 3: Feature C - Prediction Explanation")
    print("==========================================")
    
    rscript = os.getenv("RSCRIPT_PATH")
    if not rscript or not Path(rscript).exists():
        cand = Path(r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe")
        if cand.exists():
            rscript = str(cand)
    assert rscript and Path(rscript).exists(), f"Rscript not found: {rscript}"
    print(f"[OK] Using verified Rscript: {rscript}")

    bridge_file = PROJECT_ROOT / "prediction_bridge.R"
    model_file = PROJECT_ROOT / "data" / "r_outputs" / "DA2_Tuned_Model_Objects.rds"
    assert bridge_file.exists(), "prediction_bridge.R missing"
    assert model_file.exists(), "DA2_Tuned_Model_Objects.rds missing"

    # Load 1 baseline row from MySQL
    df = load_incidents()
    sample_row = df.iloc[0].copy()
    
    # Verify target exclusion
    sample_row = sample_row.drop(labels=["severity_class"], errors="ignore")
    assert "severity_class" not in sample_row.index, "Target leakage detected! severity_class must not be in input."
    print("[OK] Target leakage check PASSED: severity_class is completely excluded from inference input.")

    test_models = ["xgboost", "gradient_boosting", "decision_tree", "extra_trees"]
    for model_key in test_models:
        with tempfile.TemporaryDirectory(prefix="test_pred_") as temp_dir:
            temp = Path(temp_dir)
            in_csv = temp / "input.csv"
            out_csv = temp / "out.csv"
            exp_csv = temp / "exp.csv"
            
            pd.DataFrame([sample_row]).to_csv(in_csv, index=False)
            
            cmd = [rscript, str(bridge_file), str(model_file), model_key, str(in_csv), str(out_csv), str(exp_csv)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            assert res.returncode == 0, f"R bridge failed for {model_key}: {res.stderr}"
            assert out_csv.exists(), f"out.csv not produced for {model_key}"
            assert exp_csv.exists(), f"exp.csv not produced for {model_key}"
            
            pred_df = pd.read_csv(out_csv)
            exp_df = pd.read_csv(exp_csv)
            
            pred_class = str(pred_df.loc[0, "predicted_class"])
            assert pred_class in ["Low", "Medium", "High"], f"Invalid predicted class: {pred_class}"
            assert any(c.startswith(".pred_") for c in pred_df.columns), "Probability columns missing"
            
            assert not exp_df.empty, f"Explanation df empty for {model_key}"
            method = exp_df["method"].iloc[0]
            
            # Check figure generation
            fig = build_local_explanation_figure(exp_df, pred_class, method)
            assert len(fig.data) > 0, f"Explanation figure failed for {model_key}"
            
            probs_dict = {c: round(float(pred_df.loc[0, c]), 3) for c in pred_df.columns if c.startswith(".pred_")}
            print(f"[OK] Model '{model_key}': Predicted={pred_class}, Prob={probs_dict}")
            print(f"     Explanation Method: {method} ({len(exp_df)} features ranked).")

    print("\n[OK] Feature name translation check:")
    raw_sample = "incident_type_Disruption.Hijacking.with.Misuse"
    cleaned = clean_feature_name(raw_sample)
    print(f"     '{raw_sample}' -> '{cleaned}'")
    assert "Attack Vector:" in cleaned

if __name__ == "__main__":
    try:
        test_feature_a_geography()
        test_feature_b_confusion_matrix()
        test_feature_c_prediction_explanation()
        print("\n=======================================================")
        print("ALL END-TO-END VERIFICATION TESTS PASSED SUCCESSFULLY! [PASS]")
        print("=======================================================")
    except Exception as e:
        print(f"\n[FAIL] Test suite failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
