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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
np.random.seed(42)

N_PERMS = 50 

def get_paths():
    base_path = os.path.dirname(os.path.abspath(__file__)) 
    return {
        "input": os.path.join(base_path, "finalized_corr_shrinkage.pkl"),
        "output": os.path.join(base_path, "Weighted_FRC_Final_Analysis"),
        "aggregate": os.path.join(base_path, "Weighted_FRC_Final_Analysis", "Summary_All_Models"),
        "n_jobs": -1  
    }

PATHS = get_paths()

def compute_node_wfrc_vectorized(adj_mask, weight_matrix):
    weights = np.abs(weight_matrix) * adj_mask
    strength = np.sum(weights, axis=1)
    if np.sum(strength) == 0: return np.zeros_like(strength)
    neighbor_strength_sum = adj_mask.dot(strength)
    return 4 * strength - (strength ** 2) - neighbor_strength_sum

def get_separated_weighted_frc_features(matrices, threshold=0.2):
    n_subj = matrices.shape[0]
    f_pos, f_neg, f_combined = [], [], []
    for i in range(n_subj):
        mat = matrices[i]
        adj_pos = (mat > threshold).astype(float)
        np.fill_diagonal(adj_pos, 0)
        wfrc_pos = compute_node_wfrc_vectorized(adj_pos, mat)
        
        adj_neg = (mat < -threshold).astype(float)
        np.fill_diagonal(adj_neg, 0)
        wfrc_neg = compute_node_wfrc_vectorized(adj_neg, mat)
        
        f_pos.append(wfrc_pos)
        f_neg.append(wfrc_neg)
        f_combined.append(np.concatenate([wfrc_pos, wfrc_neg])) 
        
    return np.array(f_pos), np.array(f_neg), np.array(f_combined)

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
    plt.hist(perm_accs, bins=10, color='lightgray', edgecolor='black', alpha=0.7, label='Null Distribution')
    plt.axvline(true_acc, color='red', linestyle='dashed', linewidth=2, label=f'Observed Acc: {true_acc:.3f}')
    plt.title(f"{model_id} - Permutation Test\n{N_PERMS} Permutations | p-value ≈ {p_val:.3f}")
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
        print(f"✅ Aggregate results synchronized at {agg_dir}")

def block_permute(y, site_ids):
    y_perm = np.copy(y)
    for site in np.unique(site_ids):
        mask = (site_ids == site)
        y_perm[mask] = np.random.permutation(y[mask])
    return y_perm

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
        best_params_per_fold.append(grid.best_params_)

        preds = model.predict(X_te)
        probs = model.predict_proba(X_te)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_te)

        auc = roc_auc_score(y_te, probs)
        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds, zero_division=0)
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

        print(f"  --> Fold {fold}/5 | ACC: {acc:.3f} | AUC: {auc:.3f} | SENS: {sens:.3f} | SPEC: {spec:.3f} | Time: {time.time()-fold_start:.1f}s")

        unique_sites = np.unique(site_te)
        for site in unique_sites:
            mask = (site_te == site)
            if np.sum(mask) > 0:
                y_s = y_te[mask]
                p_s = preds[mask]
                probs_s = probs[mask]
                
                cm_s = confusion_matrix(y_s, p_s, labels=[0, 1])
                tn_s, fp_s, fn_s, tp_s = cm_s.ravel()
                
                if len(np.unique(y_s)) > 1:
                    auc_s = roc_auc_score(y_s, probs_s)
                else:
                    auc_s = np.nan
                
                site_metrics_list.append({
                    "Fold": fold, "Site": site, "N_Samples": len(y_s),
                    "ACC": accuracy_score(y_s, p_s),
                    "AUC": auc_s,
                    "F1": f1_score(y_s, p_s, zero_division=0),
                    "Sens": tp_s/(tp_s+fn_s) if (tp_s+fn_s)>0 else 0,
                    "Spec": tn_s/(tn_s+fp_s) if (tn_s+fp_s)>0 else 0,
                    "TN": tn_s, "FP": fp_s, "FN": fn_s, "TP": tp_s
                })

        del X_tr, X_te, y_tr, y_te, model, preds, probs
        gc.collect()

    final_metrics = {
        "auc_mean": np.mean(metrics["auc"]), "auc_std": np.std(metrics["auc"]),
        "acc_mean": np.mean(metrics["acc"]), "acc_std": np.std(metrics["acc"]),
        "f1_mean": np.mean(metrics["f1"]), "f1_std": np.std(metrics["f1"]),
        "sens_mean": np.mean(metrics["sens"]), "sens_std": np.std(metrics["sens"]),
        "spec_mean": np.mean(metrics["spec"]), "spec_std": np.std(metrics["spec"]),
        "cm_total": np.sum(metrics["cms"], axis=0),
    }
    for k in ["auc", "acc", "f1", "sens", "spec"]: final_metrics[f"{k}_folds"] = metrics[k]

    return final_metrics, best_params_per_fold, pd.DataFrame(site_metrics_list)

def run_permutation_test(X, y, strat_labels, site_ids, pipeline, param_grid):
    outer_cv = StratifiedKFold(n_splits=3, shuffle=True)
    inner_cv = StratifiedKFold(n_splits=2, shuffle=True)
    
    perm_acc_list = []
    
    for p in range(N_PERMS):
        perm_start = time.time()
        y_perm = block_permute(y, site_ids)
        strat_perm = np.array([f"{s}_{l}" for s, l in zip(site_ids, y_perm)])
        strat_perm = np.array([s if pd.Series(strat_perm).value_counts()[s] >= 3 else "Other" for s in strat_perm])
        
        fold_accs = []
        for tr_idx, te_idx in outer_cv.split(X, strat_perm):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y_perm[tr_idx], y_perm[te_idx]
            
            grid = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring="accuracy", n_jobs=PATHS["n_jobs"])
            grid.fit(X_tr, y_tr)
            fold_accs.append(accuracy_score(y_te, grid.predict(X_te)))
            
        perm_acc_list.append(np.mean(fold_accs))
        if (p+1) % 10 == 0:
            avg_time = time.time() - perm_start
            print(f"      [Perm {p+1:02d}/{N_PERMS}] Completed. (~{avg_time:.1f}s per iteration)")
            
    return perm_acc_list


def save_results(metrics, best_params, site_df, p_val, model_name, folder):
    os.makedirs(folder, exist_ok=True)

    # Save Mean and Std values
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

    # Save Fold-level metrics
    fold_rows = []
    for i in range(5):
        fold_rows.append({
            "Fold": i+1, "AUC": metrics["auc_folds"][i], "ACC": metrics["acc_folds"][i],
            "F1": metrics["f1_folds"][i], "Sens": metrics["sens_folds"][i],
            "Spec": metrics["spec_folds"][i], "Best_Params": str(best_params[i])
        })
    pd.DataFrame(fold_rows).to_csv(os.path.join(folder, f"{model_name}_folds_summary.csv"), index=False)

    # Save Site-level metrics
    if not site_df.empty:
        site_df.to_csv(os.path.join(folder, f"{model_name}_site_folds_detail.csv"), index=False)
        
        site_summary = site_df.groupby("Site").agg(
            Total_Samples=("N_Samples", "sum"),
            ACC_Mean=("ACC", "mean"), ACC_Std=("ACC", "std"),
            AUC_Mean=("AUC", lambda x: np.nanmean(x) if x.notna().any() else np.nan),
            AUC_Std=("AUC", lambda x: np.nanstd(x) if x.notna().any() else np.nan),
            F1_Mean=("F1", "mean"),
            Sens_Mean=("Sens", "mean"),
            Spec_Mean=("Spec", "mean"),
            TN_Total=("TN", "sum"), FP_Total=("FP", "sum"), 
            FN_Total=("FN", "sum"), TP_Total=("TP", "sum")
        ).reset_index()
        site_summary.to_csv(os.path.join(folder, f"{model_name}_site_summary_final.csv"), index=False)


def run_comprehensive_study(atlas_name="cc200"):
    print("\nLoading dataset...")
    with open(PATHS["input"], "rb") as f: df = pickle.load(f)

    df.columns = df.columns.astype(str).str.strip()
    if df.index.name in ["ASD", "SITE_ID", "DX_GROUP"]: df = df.reset_index()
    if "ASD" not in df.columns:
        if "DX_GROUP" in df.columns: df.rename(columns={"DX_GROUP": "ASD"}, inplace=True)
        elif "asd" in df.columns: df.rename(columns={"asd": "ASD"}, inplace=True)
    if "SITE_ID" not in df.columns:
        if "Site" in df.columns: df.rename(columns={"Site": "SITE_ID"}, inplace=True)
        elif "site_id" in df.columns: df.rename(columns={"site_id": "SITE_ID"}, inplace=True)

    df = df.dropna(subset=["ASD", "SITE_ID"])
    matrices = np.array(df[f"corr_{atlas_name}"].tolist(), dtype=np.float64)
    y = df["ASD"].values
    site_ids = df["SITE_ID"].values

    df["strat"] = df["SITE_ID"].astype(str) + "_" + df["ASD"].astype(str)
    counts = df["strat"].value_counts()
    strat_labels = np.array([s if counts[s] >= 5 else "Other" for s in df["strat"]])
    
    del df; gc.collect()

    thresholds = [0.1, 0.2, 0.3]
    
    clf_grids = {
        'LogReg': {
            'pipeline': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=2000))]),
            'grid': {'clf__C': [0.001, 0.1, 1, 10, 100]}
        },
        'LinSVM': {
            'pipeline': Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear', class_weight='balanced', probability=True, max_iter=2000))]),
            'grid': {'clf__C': [0.001, 0.1, 1, 10, 100]}
        },
        'RBFSVM': {
            'pipeline': Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', class_weight='balanced', probability=True, max_iter=2000))]),
            'grid': {'clf__C': [0.1, 1, 10, 100]}
        }
    }

    completed_models = []

    for thresh in thresholds:
        print(f"\n{'='*60}\n EXTRACTING FEATURES (Threshold: {thresh})\n{'='*60}")
        X_pos, X_neg, X_combined = get_separated_weighted_frc_features(matrices, threshold=thresh)
        feature_sets = {'Pos': X_pos, 'Neg': X_neg, 'Comb': X_combined}

        for feat_name, X_data in feature_sets.items():
            for clf_name, config in clf_grids.items():
                
                model_id = f"Thr{thresh}_{feat_name}_{clf_name}"
                out_dir = os.path.join(PATHS["output"], model_id)
                os.makedirs(out_dir, exist_ok=True)
                print(f"\n--- 🚀 RUNNING: {model_id} ---")

                metrics, best_params, site_df = run_nested_cv(X_data, y, strat_labels, site_ids, config["pipeline"], config["grid"])
                true_acc = metrics["acc_mean"]
                print(f"True Run Finished! Overall ACC: {true_acc:.3f} ± {metrics['acc_std']:.3f}\n")
                
                print(f"--- Starting {N_PERMS} Permutations (3-Fold Speed CV) ---")
                perm_accs = run_permutation_test(X_data, y, strat_labels, site_ids, config["pipeline"], config["grid"])
                
                p_val = (np.sum(np.array(perm_accs) >= true_acc) + 1) / (N_PERMS + 1)
                print(f"Permutations done. Final P-value: {p_val:.3f}")
                
                save_results(metrics, best_params, site_df, p_val, model_id, out_dir)
                save_confusion_plot(metrics["cm_total"], model_id, metrics["sens_mean"], metrics["spec_mean"], out_dir)
                save_permutation_plot(true_acc, perm_accs, p_val, model_id, out_dir)
                
                completed_models.append(model_id)
                print(f"{model_id} saved locally.")

                generate_aggregate_summary(completed_models)

    print("\nALL WFRC SWEEPS COMPLETED!")

if __name__ == "__main__":
    run_comprehensive_study()