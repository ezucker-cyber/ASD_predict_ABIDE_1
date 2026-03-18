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
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from pyriemann.tangentspace import TangentSpace

warnings.filterwarnings("ignore")
np.random.seed(42)

N_PERMS = 50 

def get_paths():
    
    base_path = os.path.dirname(os.path.abspath(__file__)) 

    
    return {
        "input": os.path.join(base_path, "finalized_corr_shrinkage.pkl"),
        "output": os.path.join(base_path, "MultiAtlas_Ensemble_Soft_Meta"),
        "aggregate": os.path.join(base_path, "MultiAtlas_Ensemble_Optimized"),
        "n_jobs": -1  
    }

PATHS = get_paths()

os.makedirs(PATHS["output"], exist_ok=True)
os.makedirs(PATHS["aggregate"], exist_ok=True)

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

def block_permute(y, site_ids):
    y_perm = np.copy(y)
    for site in np.unique(site_ids):
        mask = (site_ids == site)
        y_perm[mask] = np.random.permutation(y[mask])
    return y_perm

def get_metrics_dict(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "sens": tp/(tp+fn) if (tp+fn)>0 else 0,
        "spec": tn/(tn+fp) if (tn+fp)>0 else 0,
        "cm": cm
    }

def run_all_in_one_eval(X_dict, y, strat_labels, site_ids, atlases, clf_base, param_grid, base_name):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    model_keys = [f"Single_{base_name}_{a.upper()}" for a in atlases] + [f"Ensemble_{base_name}_Soft", f"Ensemble_{base_name}_Meta"]
    results = {k: {"auc": [], "acc": [], "f1": [], "sens": [], "spec": [], "cms": [], "best_params": []} for k in model_keys}
    site_metrics_list = []
    
    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_dict[atlases[0]], strat_labels), 1):
        fold_start = time.time()
        y_tr, y_te = y[tr_idx], y[te_idx]
        site_te = site_ids[te_idx]
        
        ts_features_tr, ts_features_te = [], []
        fold_test_probs = []
        best_estimators = []

        for i, atlas in enumerate(atlases):
            ts = TangentSpace(metric='riemann')
            X_ts_tr = ts.fit_transform(X_dict[atlas][tr_idx])
            X_ts_te = ts.transform(X_dict[atlas][te_idx])
            ts_features_tr.append(X_ts_tr)
            ts_features_te.append(X_ts_te)
            
            grid = GridSearchCV(clf_base, param_grid, cv=list(inner_cv.split(X_ts_tr, strat_labels[tr_idx])), scoring="roc_auc", n_jobs=PATHS["n_jobs"])
            grid.fit(X_ts_tr, y_tr)
            
            best_model = grid.best_estimator_
            best_estimators.append(best_model)
            probs = best_model.predict_proba(X_ts_te)[:, 1]
            fold_test_probs.append(probs)
        
            m = get_metrics_dict(y_te, probs)
            key = f"Single_{base_name}_{atlas.upper()}"
            for met in ["auc", "acc", "f1", "sens", "spec"]: results[key][met].append(m[met])
            results[key]["cms"].append(m["cm"])
            results[key]["best_params"].append(grid.best_params_)
            
            for site in np.unique(site_te):
                mask = (site_te == site)
                site_metrics_list.append({"Model": key, "Fold": fold, "Site": site, "N": len(y_te[mask]), "ACC": accuracy_score(y_te[mask], (probs[mask]>=0.5).astype(int))})

        soft_probs = np.mean(fold_test_probs, axis=0)
        m_s = get_metrics_dict(y_te, soft_probs)
        k_s = f"Ensemble_{base_name}_Soft"
        for met in ["auc", "acc", "f1", "sens", "spec"]: results[k_s][met].append(m_s[met])
        results[k_s]["cms"].append(m_s["cm"])
        results[k_s]["best_params"].append("Averaged")
        for site in np.unique(site_te):
            mask = (site_te == site)
            site_metrics_list.append({"Model": k_s, "Fold": fold, "Site": site, "N": len(y_te[mask]), "ACC": accuracy_score(y_te[mask], (soft_probs[mask]>=0.5).astype(int))})

        meta_X_tr = np.zeros((len(y_tr), len(atlases)))
        for i in range(len(atlases)):
            meta_X_tr[:, i] = cross_val_predict(best_estimators[i], ts_features_tr[i], y_tr, cv=3, method='predict_proba', n_jobs=PATHS["n_jobs"])[:, 1]
        
        meta_clf = LogisticRegression(class_weight='balanced', penalty='l2')
        meta_clf.fit(meta_X_tr, y_tr)
        meta_probs = meta_clf.predict_proba(np.column_stack(fold_test_probs))[:, 1]
        
        m_m = get_metrics_dict(y_te, meta_probs)
        k_m = f"Ensemble_{base_name}_Meta"
        for met in ["auc", "acc", "f1", "sens", "spec"]: results[k_m][met].append(m_m[met])
        results[k_m]["cms"].append(m_m["cm"])
        results[k_m]["best_params"].append(str(meta_clf.coef_[0]))
        for site in np.unique(site_te):
            mask = (site_te == site)
            site_metrics_list.append({"Model": k_m, "Fold": fold, "Site": site, "N": len(y_te[mask]), "ACC": accuracy_score(y_te[mask], (meta_probs[mask]>=0.5).astype(int))})

        print(f"  --> Fold {fold}/5 Completed | Time: {time.time()-fold_start:.1f}s")

    final_metrics = {}
    for k in model_keys:
        final_metrics[k] = {
            "auc_mean": np.mean(results[k]["auc"]), "auc_std": np.std(results[k]["auc"]),
            "acc_mean": np.mean(results[k]["acc"]), "acc_std": np.std(results[k]["acc"]),
            "f1_mean": np.mean(results[k]["f1"]), "f1_std": np.std(results[k]["f1"]),
            "sens_mean": np.mean(results[k]["sens"]), "sens_std": np.std(results[k]["sens"]),
            "spec_mean": np.mean(results[k]["spec"]), "spec_std": np.std(results[k]["spec"]),
            "cm_total": np.sum(results[k]["cms"], axis=0),
            "fold_data": results[k]
        }
    return final_metrics, pd.DataFrame(site_metrics_list)

def run_perm_block(X_dict, y_perm, site_ids, atlases, clf_base, param_grid, base_name):
    strat_labels_perm = np.array([f"{s}_{y}" for s, y in zip(site_ids, y_perm)])
    counts = pd.Series(strat_labels_perm).value_counts()
    strat_labels_perm = np.array([s if counts[s] >= 5 else "Other" for s in strat_labels_perm])

    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=np.random.randint(10000))
    model_keys = [f"Single_{base_name}_{a.upper()}" for a in atlases] + [f"Ensemble_{base_name}_Soft", f"Ensemble_{base_name}_Meta"]
    fold_accs = {k: [] for k in model_keys}

    for tr_idx, te_idx in outer_cv.split(X_dict[atlases[0]], strat_labels_perm):
        y_tr, y_te = y_perm[tr_idx], y_perm[te_idx]
        fold_probs, best_ests, ts_tr_list = [], [], []
        
        for i, atlas in enumerate(atlases):
            ts = TangentSpace(metric='riemann')
            X_ts_tr, X_ts_te = ts.fit_transform(X_dict[atlas][tr_idx]), ts.transform(X_dict[atlas][te_idx])
            ts_tr_list.append(X_ts_tr)
            grid = GridSearchCV(clf_base, param_grid, cv=2, scoring="accuracy", n_jobs=PATHS["n_jobs"])
            grid.fit(X_ts_tr, y_tr)
            p = grid.predict_proba(X_ts_te)[:, 1]
            fold_probs.append(p)
            best_ests.append(grid.best_estimator_)
            fold_accs[f"Single_{base_name}_{atlas.upper()}"].append(accuracy_score(y_te, (p>=0.5).astype(int)))

        fold_accs[f"Ensemble_{base_name}_Soft"].append(accuracy_score(y_te, (np.mean(fold_probs, axis=0)>=0.5).astype(int)))
        
        meta_X_tr = np.zeros((len(y_tr), len(atlases)))
        for i in range(len(atlases)):
            meta_X_tr[:, i] = cross_val_predict(best_ests[i], ts_tr_list[i], y_tr, cv=2, method='predict_proba', n_jobs=PATHS["n_jobs"])[:, 1]
        m_clf = LogisticRegression(class_weight='balanced').fit(meta_X_tr, y_tr)
        m_p = m_clf.predict_proba(np.column_stack(fold_probs))[:, 1]
        fold_accs[f"Ensemble_{base_name}_Meta"].append(accuracy_score(y_te, (m_p>=0.5).astype(int)))

    return {k: np.mean(v) for k, v in fold_accs.items()}

def save_all_artifacts(metrics, site_df, p_vals, perms_dict, base_name):
    for k, data in metrics.items():
        folder = os.path.join(PATHS["output"], k)
        os.makedirs(folder, exist_ok=True)
        pd.DataFrame([{
            "Model": k, "P_Val": round(p_vals[k], 4), "ACC": round(data["acc_mean"], 3), "AUC": round(data["auc_mean"], 3),
            "Sens": round(data["sens_mean"], 3), "Spec": round(data["spec_mean"], 3),
            "TN": data["cm_total"][0,0], "FP": data["cm_total"][0,1], "FN": data["cm_total"][1,0], "TP": data["cm_total"][1,1]
        }]).to_csv(os.path.join(folder, f"{k}_summary.csv"), index=False)

        fd = data["fold_data"]
        pd.DataFrame({
            "Fold": np.arange(1,6), "ACC": fd["acc"], "AUC": fd["auc"], "Params": fd["best_params"]
        }).to_csv(os.path.join(folder, f"{k}_folds_summary.csv"), index=False)
        s_k = site_df[site_df["Model"] == k]
        s_k.to_csv(os.path.join(folder, f"{k}_site_folds_detail.csv"), index=False)
        s_k.groupby("Site").agg(N_Total=("N", "sum"), ACC_Avg=("ACC", "mean"), ACC_Std=("ACC", "std")).to_csv(os.path.join(folder, f"{k}_site_summary_final.csv"))

        save_confusion_plot(data["cm_total"], k, data["sens_mean"], data["spec_mean"], folder)
        save_permutation_plot(data["acc_mean"], perms_dict[k], p_vals[k], k, folder)


def run_study():
    print("Loading Data")
    if not os.path.exists(PATHS["input"]):
        print(f" Error: Data file not found at {PATHS['input']}")
        return

    with open(PATHS["input"], "rb") as f: df = pickle.load(f)
    
    df.columns = df.columns.astype(str).str.strip()
    if "DX_GROUP" in df.columns: df.rename(columns={"DX_GROUP": "ASD"}, inplace=True)
    if "Site" in df.columns: df.rename(columns={"Site": "SITE_ID"}, inplace=True)
    df = df.dropna(subset=["ASD", "SITE_ID"])

    atlases = ["cc200", "aal", "dos160"] 
    X_dict = {a: np.ascontiguousarray(np.stack(df[f"corr_{a}"].values), dtype=np.float64) for a in atlases}
    y, site_ids = df["ASD"].values, df["SITE_ID"].values
    df["strat"] = df["SITE_ID"].astype(str) + "_" + df["ASD"].astype(str)
    counts = df["strat"].value_counts()
    strat_labels = np.array([s if counts[s] >= 5 else "Other" for s in df["strat"]])

    classifiers = {
        "TS_Riemann_LinSVM": {
            
            "clf": SVC(kernel='linear', class_weight='balanced', probability=True, max_iter=2000),
            "grid": {"C": [0.001, 0.1, 1, 10, 100]}
        },
        "TS_Riemann_RBFSVM": {
            "clf": SVC(kernel='rbf', class_weight='balanced', probability=True, max_iter=2000),
            "grid": {"C": [0.1, 1, 10, 100]}
        },
        "TS_Riemann_LogReg": {
            "clf": LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=2000),
            "grid": {"C": [0.001, 0.1, 1, 10, 100]}
        }

    }

    for clf_name, config in classifiers.items():
        print(f"\n STARTING BLOCK: {clf_name}")
        metrics, site_df = run_all_in_one_eval(X_dict, y, strat_labels, site_ids, atlases, config["clf"], config["grid"], clf_name)
        
        print(f"--- Starting {N_PERMS} Permutations ---")
        perm_accs = {k: [] for k in metrics.keys()}
        for p in range(N_PERMS):
            y_p = block_permute(y, site_ids)
            p_res = run_perm_block(X_dict, y_p, site_ids, atlases, config["clf"], config["grid"], clf_name)
            for k in metrics.keys(): perm_accs[k].append(p_res[k])
            if (p+1)%10==0: print(f"  [Perm {p+1:03d}/{N_PERMS}] Completed.")

        p_vals = {k: (np.sum(np.array(perm_accs[k]) >= metrics[k]["acc_mean"]) + 1) / (N_PERMS + 1) for k in metrics.keys()}
        save_all_artifacts(metrics, site_df, p_vals, perm_accs, clf_name)

    print("\n ALL STUDIES COMPLETED!")

if __name__ == "__main__":
    run_study()