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
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings("ignore")
np.random.seed(42)

N_PERMS = 50  

def get_paths():
    base_path = os.path.dirname(os.path.abspath(__file__)) 
    return {
        "input": os.path.join(base_path, "finalized_corr_shrinkage.pkl"),
        "output": os.path.join(base_path, "Advanced_Geometric_Baselines"),
        "aggregate": os.path.join(base_path, "Advanced_Geometric_Baselines_Optimized", "Summary_All_ClassicalML_Models"),
        "n_jobs": -1 
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

def save_permutation_plot(true_acc, perm_accs, p_val, model_id, folder):
    plt.figure(figsize=(7,5))
    plt.hist(perm_accs, bins=15, color='lightgray', edgecolor='black', alpha=0.7, label='Null Distribution')
    plt.axvline(true_acc, color='red', linestyle='dashed', linewidth=2, label=f'Observed Acc: {true_acc:.3f}')
    plt.title(f"{model_id} - Permutation Test\n{N_PERMS} Permutations | p-value = {p_val:.3f}")
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"perm_hist_{model_id}.png"))
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
            for suffix in [f"cm_{model_name}.png", f"perm_hist_{model_name}.png"]:
                src = os.path.join(model_dir, suffix)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(agg_dir, f"{model_name}_{suffix}"))
    
    if all_summaries:
        master_df = pd.concat(all_summaries, ignore_index=True)
        master_df.to_csv(os.path.join(agg_dir, "MASTER_COMPARISON_SUMMARY.csv"), index=False)
        print(f"Aggregate results and charts saved to {agg_dir}")



def block_permute(y, site_ids):
    """Permutes labels independently within each site."""
    y_perm = np.copy(y)
    for site in np.unique(site_ids):
        mask = (site_ids == site)
        y_perm[mask] = np.random.permutation(y[mask])
    return y_perm

def get_metrics(y_true, y_prob=None, y_pred_hard=None):
    if y_pred_hard is None:
        y_pred_hard = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_hard, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "auc": roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan,
        "acc": accuracy_score(y_true, y_pred_hard),
        "f1": f1_score(y_true, y_pred_hard) if (tp+fp)>0 else 0,
        "sens": tp/(tp+fn) if (tp+fn)>0 else 0,
        "spec": tn/(tn+fp) if (tn+fp)>0 else 0,
        "cm": cm
    }

def run_all_in_one_evaluation(X, y, strat_labels, site_ids, classifiers):
    """Computes Tangent Space once per fold and shares it across all classifiers."""
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    model_keys = list(classifiers.keys())
    results = {k: {"auc": [], "acc": [], "f1": [], "sens": [], "spec": [], "cms": []} for k in model_keys}
    best_params_log = {k: [] for k in model_keys}
    site_metrics_list = []
    
    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X, strat_labels), 1):
        fold_start = time.time()
        X_raw_tr, X_raw_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        site_te = site_ids[te_idx]
        
        ts = TangentSpace(metric='riemann')
        X_ts_tr = ts.fit_transform(X_raw_tr)
        X_ts_te = ts.transform(X_raw_te)
        
        for clf_name, config in classifiers.items():
            grid = GridSearchCV(config["clf"], config["grid"], cv=list(inner_cv.split(X_ts_tr, strat_labels[tr_idx])), scoring="roc_auc", n_jobs=PATHS["n_jobs"])
            grid.fit(X_ts_tr, y_tr)
            
            best_model = grid.best_estimator_
            best_params_log[clf_name].append(grid.best_params_)
            
            preds = best_model.predict(X_ts_te)
            probs = best_model.predict_proba(X_ts_te)[:, 1] if hasattr(best_model, "predict_proba") else best_model.decision_function(X_ts_te)
            
            m = get_metrics(y_te, y_prob=probs, y_pred_hard=preds)
            for metric in ["auc", "acc", "f1", "sens", "spec"]: results[clf_name][metric].append(m[metric])
            results[clf_name]["cms"].append(m["cm"])
            
            for site in np.unique(site_te):
                mask = (site_te == site)
                site_metrics_list.append({
                    "Model": clf_name, "Fold": fold, "Site": site, "N_Samples": len(y_te[mask]),
                    "ACC": accuracy_score(y_te[mask], preds[mask]),
                    "AUC": roc_auc_score(y_te[mask], probs[mask]) if len(np.unique(y_te[mask])) > 1 else np.nan
                })
                
        print(f"  --> Fold {fold}/5 Completed | Time: {time.time()-fold_start:.1f}s")
        del X_raw_tr, X_raw_te, X_ts_tr, X_ts_te
        gc.collect()

    final_metrics = {}
    for k in model_keys:
        final_metrics[k] = {
            "auc_mean": np.mean(results[k]["auc"]), "auc_std": np.std(results[k]["auc"]),
            "acc_mean": np.mean(results[k]["acc"]), "acc_std": np.std(results[k]["acc"]),
            "f1_mean": np.mean(results[k]["f1"]), "f1_std": np.std(results[k]["f1"]),
            "sens_mean": np.mean(results[k]["sens"]), "sens_std": np.std(results[k]["sens"]),
            "spec_mean": np.mean(results[k]["spec"]), "spec_std": np.std(results[k]["spec"]),
            "cm_total": np.sum(results[k]["cms"], axis=0)
        }
        for metric in ["auc", "acc", "f1", "sens", "spec"]: final_metrics[k][f"{metric}_folds"] = results[k][metric]
            
    return final_metrics, best_params_log, pd.DataFrame(site_metrics_list)

def run_all_in_one_perm(X, y_perm, site_ids, classifiers):
    strat_labels_perm = np.array([f"{s}_{y}" for s, y in zip(site_ids, y_perm)])
    counts = pd.Series(strat_labels_perm).value_counts()
    strat_labels_perm = np.array([s if counts[s] >= 5 else "Other" for s in strat_labels_perm])

    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=np.random.randint(10000))
    inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=np.random.randint(10000))
    
    fold_accs = {k: [] for k in classifiers.keys()}

    for tr_idx, te_idx in outer_cv.split(X, strat_labels_perm):
        X_raw_tr, X_raw_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y_perm[tr_idx], y_perm[te_idx]
        
        ts = TangentSpace(metric='riemann')
        X_ts_tr = ts.fit_transform(X_raw_tr)
        X_ts_te = ts.transform(X_raw_te)
        
        for clf_name, config in classifiers.items():
            grid = GridSearchCV(config["clf"], config["grid"], cv=inner_cv, scoring="accuracy", n_jobs=PATHS["n_jobs"])
            grid.fit(X_ts_tr, y_tr)
            preds = grid.predict(X_ts_te)
            fold_accs[clf_name].append(accuracy_score(y_te, preds))

    return {k: np.mean(v) for k, v in fold_accs.items()}


def save_results(metrics, best_params, site_df, p_val, model_name, folder):
    """Saves comprehensive results: overall, per-fold, site-fold, and site-summary."""
    os.makedirs(folder, exist_ok=True)
    
    summary = pd.DataFrame([{
        "Model": model_name, "P_Value_Perm": round(p_val, 4),
        "AUC_Mean": round(metrics["auc_mean"], 3), "AUC_STD": round(metrics["auc_std"], 3),
        "ACC_Mean": round(metrics["acc_mean"], 3), "ACC_STD": round(metrics["acc_std"], 3),
        "F1_Mean": round(metrics["f1_mean"], 3), "F1_STD": round(metrics["f1_std"], 3),
        "Sens_Mean": round(metrics["sens_mean"], 3), "Sens_STD": round(metrics["sens_std"], 3),
        "Spec_Mean": round(metrics["spec_mean"], 3), "Spec_STD": round(metrics["spec_std"], 3),
        "TN": metrics["cm_total"][0,0], "FP": metrics["cm_total"][0,1],
        "FN": metrics["cm_total"][1,0], "TP": metrics["cm_total"][1,1]
    }])
    summary.to_csv(os.path.join(folder, f"{model_name}_summary.csv"), index=False)

    fold_rows = []
    for i in range(5):
        fold_rows.append({
            "Fold": i+1, 
            "AUC": metrics["auc_folds"][i], "ACC": metrics["acc_folds"][i],
            "F1": metrics["f1_folds"][i], "Sens": metrics["sens_folds"][i],
            "Spec": metrics["spec_folds"][i],
            "Best_Params": str(best_params[i])
        })
    pd.DataFrame(fold_rows).to_csv(os.path.join(folder, f"{model_name}_folds_summary.csv"), index=False)

    
    if not site_df.empty:
        
        site_df.to_csv(os.path.join(folder, f"{model_name}_site_folds_detail.csv"), index=False)
        
        
        site_summary = site_df.groupby("Site").agg(
            Total_Samples_Tested=("N_Samples", "sum"),
            ACC_Mean=("ACC", "mean"), ACC_Std=("ACC", "std"),
            AUC_Mean=("AUC", lambda x: np.nanmean(x)), AUC_Std=("AUC", lambda x: np.nanstd(x))
        ).reset_index()
        site_summary.to_csv(os.path.join(folder, f"{model_name}_site_summary_final.csv"), index=False)

def run_ts_models(atlas_name="cc200"):
    print("\nLoading dataset...")
    with open(PATHS["input"], "rb") as f:
        df = pickle.load(f)

    df.columns = df.columns.astype(str).str.strip()
    if df.index.name in ["ASD", "SITE_ID", "DX_GROUP"]: df = df.reset_index()
    if "DX_GROUP" in df.columns: df.rename(columns={"DX_GROUP": "ASD"}, inplace=True)
    if "asd" in df.columns: df.rename(columns={"asd": "ASD"}, inplace=True)
    if "Site" in df.columns: df.rename(columns={"Site": "SITE_ID"}, inplace=True)
    if "site_id" in df.columns: df.rename(columns={"site_id": "SITE_ID"}, inplace=True)
    df = df.dropna(subset=["ASD", "SITE_ID"])

    X = get_full_spd_matrices(df, f"corr_{atlas_name}")
    y = df["ASD"].values
    site_ids = df["SITE_ID"].values

    df["strat"] = df["SITE_ID"].astype(str) + "_" + df["ASD"].astype(str)
    counts = df["strat"].value_counts()
    strat_labels = np.array([s if counts[s] >= 5 else "Other" for s in df["strat"]])

    del df
    gc.collect()

    classifiers = {
        "TS_Riemann_LogReg": {
            "clf": LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=2000),
            "grid": {"C": [0.001, 0.1, 1, 10, 100]}
        },
        "TS_Riemann_LinSVM": {
            "clf": LinearSVC(class_weight='balanced', dual=False, max_iter=2000),
            "grid": {"C": [0.001, 0.1, 1, 10, 100]}
        },
        "TS_Riemann_RBFSVM": {
            "clf": SVC(kernel='rbf', class_weight='balanced', probability=True, max_iter=2000),
            "grid": {"C": [0.1, 1, 10, 100]}
        }
    }

    completed_models = []
    models_to_run_keys = []
    for clf_name in classifiers.keys():
        if os.path.exists(os.path.join(PATHS["output"], clf_name, f"{clf_name}_summary.csv")):
            completed_models.append(clf_name)
        else:
            models_to_run_keys.append(clf_name)

    if not models_to_run_keys:
        print("All models already completed. Generating final aggregate summary.")
        generate_aggregate_summary(completed_models)
        return

    classifiers_to_run = {k: classifiers[k] for k in models_to_run_keys}

    print(f"\n{'='*60}\nSTARTING ALL-IN-ONE BASELINE EVALUATION\nRun queue: {models_to_run_keys}\n{'='*60}")
    
    print("--- Computing True Observations ---")
    true_metrics, best_params_log, site_metrics_df = run_all_in_one_evaluation(
        X, y, strat_labels, site_ids, classifiers_to_run
    )
    
    print(f"\n--- Starting {N_PERMS} Permutations ---")
    perm_results = {k: [] for k in classifiers_to_run.keys()}
    
    for p in range(N_PERMS):
        y_perm = block_permute(y, site_ids)
        perm_accs = run_all_in_one_perm(X, y_perm, site_ids, classifiers_to_run)
        for k in classifiers_to_run.keys(): 
            perm_results[k].append(perm_accs[k])
        
        if (p+1) % max(1, N_PERMS // 10) == 0 or p == N_PERMS - 1:
            print(f"  [Perm {p+1:03d}/{N_PERMS}] Completed.")

    for k in classifiers_to_run.keys():
        out_dir = os.path.join(PATHS["output"], k)
        true_acc = true_metrics[k]["acc_mean"]
        null_dist = perm_results[k]
        p_val = (np.sum(np.array(null_dist) >= true_acc) + 1) / (N_PERMS + 1)
        

        model_site_df = site_metrics_df[site_metrics_df["Model"] == k].drop(columns=["Model"])
        

        save_results(true_metrics[k], best_params_log[k], model_site_df, p_val, k, out_dir)
        
        save_confusion_plot(true_metrics[k]["cm_total"], k, true_metrics[k]["sens_mean"], true_metrics[k]["spec_mean"], out_dir)
        save_permutation_plot(true_acc, null_dist, p_val, k, out_dir)
        
        completed_models.append(k)
        generate_aggregate_summary(completed_models)

    print("\n ALL BASELINES COMPLETED!")

if __name__=="__main__":
    run_ts_models()