import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

def train_and_explain():
    data_path = os.path.join("backend", "data", "sdss_dr18_sample.csv")
    meta_path = os.path.join("backend", "data", "sdss_metadata.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}")

    df = pd.read_csv(data_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Derive colour indices consistently
    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z"]

    feature_cols = ["u", "g", "r", "i", "z", "redshift", "u_g", "g_r", "r_i", "i_z"]
    classes = ["STAR", "GALAXY", "QUASAR"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    X = df[feature_cols].values
    y = np.array([class_to_idx[c] for c in df["class"]])

    # 80/20 Stratified train/test split (prevent target leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ==========================================
    # 1. Random Forest Classifier (Scikit-Learn)
    # ==========================================
    print("Training Scikit-Learn RandomForestClassifier...")
    rf = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="accuracy")
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)

    rf_cm = confusion_matrix(y_test, rf_pred).tolist()
    rf_report = classification_report(y_test, rf_pred, target_names=classes, output_dict=True)

    rf_metrics = {
        "model_name": "Random Forest Classifier",
        "library": "scikit-learn 1.9",
        "holdout_accuracy": round(float(accuracy_score(y_test, rf_pred)), 4),
        "macro_precision": round(float(precision_score(y_test, rf_pred, average="macro")), 4),
        "macro_recall": round(float(recall_score(y_test, rf_pred, average="macro")), 4),
        "macro_f1": round(float(f1_score(y_test, rf_pred, average="macro")), 4),
        "roc_auc_ovr": round(float(roc_auc_score(y_test, rf_prob, multi_class="ovr")), 4),
        "cv_accuracy_mean": round(float(rf_cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(rf_cv_scores.std()), 4),
        "five_fold_cv_display": f"{rf_cv_scores.mean()*100:.2f}% ± {rf_cv_scores.std()*100:.2f}%",
        "confusion_matrix": rf_cm,
        "class_metrics": {
            cls: {
                "precision": round(float(rf_report[cls]["precision"]), 4),
                "recall": round(float(rf_report[cls]["recall"]), 4),
                "f1_score": round(float(rf_report[cls]["f1-score"]), 4),
                "support": int(rf_report[cls]["support"])
            }
            for cls in classes
        }
    }

    # ==========================================
    # 2. XGBoost Classifier
    # ==========================================
    print("Training XGBoost Classifier...")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)

    xgb_cv_scores = cross_val_score(xgb, X_train, y_train, cv=cv, scoring="accuracy")
    xgb_pred = xgb.predict(X_test)
    xgb_prob = xgb.predict_proba(X_test)

    xgb_cm = confusion_matrix(y_test, xgb_pred).tolist()
    xgb_report = classification_report(y_test, xgb_pred, target_names=classes, output_dict=True)

    xgb_metrics = {
        "model_name": "XGBoost Classifier",
        "library": "xgboost 3.4",
        "holdout_accuracy": round(float(accuracy_score(y_test, xgb_pred)), 4),
        "macro_precision": round(float(precision_score(y_test, xgb_pred, average="macro")), 4),
        "macro_recall": round(float(recall_score(y_test, xgb_pred, average="macro")), 4),
        "macro_f1": round(float(f1_score(y_test, xgb_pred, average="macro")), 4),
        "roc_auc_ovr": round(float(roc_auc_score(y_test, xgb_prob, multi_class="ovr")), 4),
        "cv_accuracy_mean": round(float(xgb_cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(xgb_cv_scores.std()), 4),
        "five_fold_cv_display": f"{xgb_cv_scores.mean()*100:.2f}% ± {xgb_cv_scores.std()*100:.2f}%",
        "confusion_matrix": xgb_cm,
        "class_metrics": {
            cls: {
                "precision": round(float(xgb_report[cls]["precision"]), 4),
                "recall": round(float(xgb_report[cls]["recall"]), 4),
                "f1_score": round(float(xgb_report[cls]["f1-score"]), 4),
                "support": int(xgb_report[cls]["support"])
            }
            for cls in classes
        }
    }

    # ==========================================
    # 3. Model-Specific SHAP Explainers
    # ==========================================
    print("Computing Model-Specific SHAP Explainers...")
    # TreeExplainer for Random Forest
    rf_explainer = shap.TreeExplainer(rf)
    # TreeExplainer for XGBoost
    xgb_explainer = shap.TreeExplainer(xgb)

    # Compute global SHAP values on test sample
    sample_X_test = X_test[:100]
    rf_shap_vals = rf_explainer.shap_values(sample_X_test)
    xgb_shap_vals = xgb_explainer.shap_values(sample_X_test)

    # For multi-class, compute mean absolute SHAP per feature across classes
    def get_global_shap(shap_arr):
        if isinstance(shap_arr, list): # list of 3 arrays (one per class)
            stacked = np.mean([np.abs(c) for c in shap_arr], axis=0) # shape (N, D)
            mean_abs = np.mean(stacked, axis=0)
        else: # 3D array (N, D, C)
            mean_abs = np.mean(np.abs(shap_arr), axis=(0, 2)) if shap_arr.ndim == 3 else np.mean(np.abs(shap_arr), axis=0)
        
        tot = float(np.sum(mean_abs)) or 1.0
        return [
            {"feature": f, "mean_abs_shap": round(float(v), 4), "importance_pct": round(float(v / tot * 100), 1)}
            for f, v in sorted(zip(feature_cols, mean_abs), key=lambda x: x[1], reverse=True)
        ]

    rf_global_shap = get_global_shap(rf_shap_vals)
    xgb_global_shap = get_global_shap(xgb_shap_vals)

    print(f"RF Holdout Accuracy: {rf_metrics['holdout_accuracy']}, Macro F1: {rf_metrics['macro_f1']}")
    print(f"XGB Holdout Accuracy: {xgb_metrics['holdout_accuracy']}, Macro F1: {xgb_metrics['macro_f1']}")

    # Save comprehensive bundle
    os.makedirs(os.path.join("backend", "models"), exist_ok=True)
    bundle = {
        "version": "2.4.0-sdss",
        "dataset_metadata": metadata,
        "feature_names": feature_cols,
        "classes": classes,
        "models": {
            "rf": rf,
            "xgb": xgb
        },
        "explainers": {
            "rf": rf_explainer,
            "xgb": xgb_explainer
        },
        "metrics": {
            "rf": rf_metrics,
            "xgb": xgb_metrics
        },
        "global_shap": {
            "rf": rf_global_shap,
            "xgb": xgb_global_shap
        },
        "split_strategy": "80/20 Stratified (1,200 train / 300 test), 5-Fold Stratified CV"
    }

    models_path = os.path.join("backend", "models", "sdss_models.joblib")
    joblib.dump(bundle, models_path)
    # Also update legacy sdss_rf.joblib for backwards compatibility
    legacy_bundle = {
        "model": rf,
        "feature_names": feature_cols,
        "classes": classes,
        "metrics": rf_metrics
    }
    joblib.dump(legacy_bundle, os.path.join("backend", "models", "sdss_rf.joblib"))

    print(f"Saved complete astronomy ML models & SHAP explainers to {models_path}")

if __name__ == "__main__":
    train_and_explain()
