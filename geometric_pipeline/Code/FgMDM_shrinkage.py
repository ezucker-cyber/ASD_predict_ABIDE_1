import numpy as np
import pandas as pd
import pickle
import os
import time
import warnings
import gc
import shutil
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from pyriemann.classification import FgMDM

warnings.filterwarnings("ignore")
np.random.seed(42)


def get_paths():
    base_path = os.path.dirname(os.path.abspath(__file__)) 
    return {
        "input": os.path.join(base_path, "finalized_corr_shrinkage.pkl"),
        "output": os.path.join(base_path, "Advanced_Geometric_Baselines"),
        "aggregate": os.path.join(base_path, "Advanced_Geometric_Baselines", "Summary_All_Models"),
        "n_jobs": 1  
    }

PATHS = get_paths()

def get_full_spd_matrices(df, col_name):
    return np.ascontiguousarray(np.stack(df[col_name].values), dtype=np.float32)

def save_confusion_plot(cm, model_id, sens, spec, folder):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Control','ASD'], yticklabels=['Control','ASD'])
    plt.title(f"{model_id}\nSens {sens:.2f} | Spec {spec:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"cm_{model_id}.png"))
    plt.close()

def generate_aggregate_summary(completed_models):
    agg_dir = PATHS["aggregate"]
    os.makedirs(agg_dir, exist_ok=True)
    all_summaries = []
    
    for model_name in completed_models:
        model_dir = os.path.join(PATHS["output"], model_name)
        summary_path = os.path.join(model_dir, f"{model_name}_summary.csv")
        
        if os.path.exists(summary_path):
            all_summaries.append(pd.read_csv(summary_path))
    
    if all_summaries:
        master_df = pd.concat(all_summaries, ignore_index=True)
        master_df.to_csv(os.path.join(agg_dir, "MASTER_COMPARISON_SUMMARY.csv"), index=False)

def run_nested_cv(X, y, strat_labels, site_ids, pipeline, param_grid):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    metrics = {"auc": [], "acc": [], "f1": [], "sens": [], "spec": [], "cms": []}
    site_metrics_list = []
    best_params_per_fold = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, strat_labels), 1):
        fold_start = time.time()
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        site_te = site_ids[test_idx]

        inner_splits = list(inner_cv.split(X_tr, strat_labels[train_idx]))
        grid = GridSearchCV(pipeline, param_grid, cv=inner_splits, scoring="roc_auc", n_jobs=PATHS["n_jobs"])
        grid.fit(X_tr, y_tr)
        
        model = grid.best_estimator_

        best_params_per_fold.append(str(model.named_steps['clf']))

        preds = model.predict(X_te)
        probs = model.predict_proba(X_te)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_te)

        auc = roc_auc_score(y_te, probs)
        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds)
        cm = confusion_matrix(y_te, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0

        metrics["auc"].append(auc)
        metrics["acc"].append(acc)
        metrics["f1"].append(f1)
        metrics["sens"].append(sens)
        metrics["spec"].append(spec)
        metrics["cms"].append(cm)

        print(f"  --> Fold {fold}/5 | ACC: {acc:.3f} | AUC: {auc:.3f}")

        unique_sites = np.unique(site_te)
        for site in unique_sites:
            mask = (site_te == site)
            y_te_s, preds_s = y_te[mask], preds[mask]
            acc_s = accuracy_score(y_te_s, preds_s)
            site_metrics_list.append({"Fold": fold, "Site": site, "N_Samples": len(y_te_s), "ACC": acc_s})

    final_metrics = {
        "auc_mean": np.mean(metrics["auc"]), "auc_std": np.std(metrics["auc"]),
        "acc_mean": np.mean(metrics["acc"]), "acc_std": np.std(metrics["acc"]),
        "f1_mean": np.mean(metrics["f1"]), "f1_std": np.std(metrics["f1"]),
        "sens_mean": np.mean(metrics["sens"]), "sens_std": np.std(metrics["sens"]),
        "spec_mean": np.mean(metrics["spec"]), "spec_std": np.std(metrics["spec"]),
        "cm_total": np.sum(metrics["cms"], axis=0),
        "auc_folds": metrics["auc"], "acc_folds": metrics["acc"],
        "f1_folds": metrics["f1"], "sens_folds": metrics["sens"], "spec_folds": metrics["spec"]
    }

    return final_metrics, best_params_per_fold, pd.DataFrame(site_metrics_list)


def save_results(metrics, best_params, site_df, model_name, folder):
    os.makedirs(folder, exist_ok=True)

    summary = pd.DataFrame([{
        "Model": model_name, 
        "AUC_Mean": round(metrics["auc_mean"], 3), "AUC_STD": round(metrics["auc_std"], 3),
        "ACC_Mean": round(metrics["acc_mean"], 3), "ACC_STD": round(metrics["acc_std"], 3),
        "F1_Mean": round(metrics["f1_mean"], 3), "F1_STD": round(metrics["f1_std"], 3),
        "Sens_Mean": round(metrics["sens_mean"], 3), "Sens_STD": round(metrics["sens_std"], 3),
        "Spec_Mean": round(metrics["spec_mean"], 3), "Spec_STD": round(metrics["spec_std"], 3),
        "TN": int(metrics["cm_total"][0,0]), "FP": int(metrics["cm_total"][0,1]),
        "FN": int(metrics["cm_total"][1,0]), "TP": int(metrics["cm_total"][1,1])
    }])
    summary.to_csv(os.path.join(folder, f"{model_name}_summary.csv"), index=False)

    fold_rows = []
    for i in range(5):
        fold_rows.append({
            "Model": model_name,
            "Fold": i + 1,
            "AUC": metrics["auc_folds"][i],
            "ACC": metrics["acc_folds"][i],
            "F1": metrics["f1_folds"][i],
            "Sens": metrics["sens_folds"][i],
            "Spec": metrics["spec_folds"][i],
            "Params": f"{{'clf': {best_params[i]}}}" 
        })
    pd.DataFrame(fold_rows).to_csv(os.path.join(folder, f"{model_name}_folds.csv"), index=False)

    if not site_df.empty:
        site_df.to_csv(os.path.join(folder, f"{model_name}_site_detail.csv"), index=False)


def run_fgmdm_only(atlas_name="cc200"):
    print("\nLoading dataset...")
    with open(PATHS["input"], "rb") as f:
        df = pickle.load(f)

    df.columns = df.columns.astype(str).str.strip()
    if "DX_GROUP" in df.columns: df.rename(columns={"DX_GROUP": "ASD"}, inplace=True)
    if "Site" in df.columns: df.rename(columns={"Site": "SITE_ID"}, inplace=True)

    X = get_full_spd_matrices(df, f"corr_{atlas_name}")
    y = (df["ASD"].values == 1).astype(int) 
    site_ids = df["SITE_ID"].values


    df["strat"] = df["SITE_ID"].astype(str) + "_" + y.astype(str)
    counts = df["strat"].value_counts()
    strat_labels = np.array([s if counts[s] >= 5 else "Other" for s in df["strat"]])

    models_to_run = {
        "FgMDM_Riemann": {
            "pipeline": Pipeline([("clf", FgMDM(metric="riemann"))]),
            "grid": {}
        }
    }

    completed_models = []

    for model_id, config in models_to_run.items():
        print(f"\n STARTING: {model_id}")
        out_dir = os.path.join(PATHS["output"], model_id)
        

        metrics, best_params, site_df = run_nested_cv(X, y, strat_labels, site_ids, config["pipeline"], config["grid"])
        
        save_results(metrics, best_params, site_df, model_id, out_dir)
        
        save_confusion_plot(metrics["cm_total"], model_id, metrics["sens_mean"], metrics["spec_mean"], out_dir)
        
        completed_models.append(model_id)
        generate_aggregate_summary(completed_models)

    print("\nPIPELINE COMPLETED!")

if __name__=="__main__":
    run_fgmdm_only()