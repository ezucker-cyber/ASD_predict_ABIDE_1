import numpy as np
import pandas as pd
import pickle
import os
import time
import warnings
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.base import invsqrtm
from neuroHarmonize import harmonizationLearn, harmonizationApply

warnings.filterwarnings("ignore")
np.random.seed(42)
N_PERMS = 50 

def get_paths():
    base_path = os.path.dirname(os.path.abspath(__file__)) 
    return {
        "input": os.path.join(base_path, "finalized_corr_shrinkage.pkl"),
        "output": os.path.join(base_path, "Scaling_Alignment"),
        "n_jobs": -1  
    }

PATHS = get_paths()
os.makedirs(PATHS["output"], exist_ok=True)

def apply_procrustes_alignment(X_train, site_train, X_test, site_test):
    X_tr_aligned, X_te_aligned = np.empty_like(X_train), np.empty_like(X_test)
    unique_sites = np.unique(site_train)
    global_invsqrt = invsqrtm(mean_covariance(X_train, metric='riemann'))
    
    for site in unique_sites:
        idx_tr = (site_train == site)
        if np.sum(idx_tr) > 0:
            site_invsqrt = invsqrtm(mean_covariance(X_train[idx_tr], metric='riemann'))
            for i in np.where(idx_tr)[0]: X_tr_aligned[i] = site_invsqrt @ X_train[i] @ site_invsqrt
            
            idx_te = (site_test == site)
            if np.sum(idx_te) > 0:
                for i in np.where(idx_te)[0]: X_te_aligned[i] = site_invsqrt @ X_test[i] @ site_invsqrt

    for site in np.unique(site_test):
        if site not in unique_sites:
            idx_te = (site_test == site)
            for i in np.where(idx_te)[0]: X_te_aligned[i] = global_invsqrt @ X_test[i] @ global_invsqrt
            
    return X_tr_aligned, X_te_aligned

def process_features(X_tr_raw, X_te_raw, y_tr, y_te, site_tr, site_te, scenario):
    if scenario == 'procrustes':
        X_tr_raw, X_te_raw = apply_procrustes_alignment(X_tr_raw, site_tr, X_te_raw, site_te)

    ts = TangentSpace(metric='riemann')
    X_tr = ts.fit_transform(X_tr_raw)
    X_te = ts.transform(X_te_raw)

    if scenario == 'neuroHarmonize':
        covars_tr = pd.DataFrame({'SITE': site_tr})
        covars_te = pd.DataFrame({'SITE': site_te})
        h_model, X_tr = harmonizationLearn(X_tr, covars_tr)
        X_te = harmonizationApply(X_te, covars_te, h_model)
    elif scenario == 'standard_scaler':
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_tr, X_te = scaler.fit_transform(X_tr), scaler.transform(X_te)
    elif scenario == 'mean_centering':
        scaler = StandardScaler(with_mean=True, with_std=False)
        X_tr, X_te = scaler.fit_transform(X_tr), scaler.transform(X_te)

    return X_tr, X_te

def save_permutation_plot(true_acc, perm_accs, model_id, folder):
    p_val = (np.sum(np.array(perm_accs) >= true_acc) + 1) / (N_PERMS + 1)
    
    plt.figure(figsize=(7,5))
    plt.hist(perm_accs, bins=10, color='lightgray', edgecolor='black', alpha=0.7, label='Null Distribution')
    plt.axvline(true_acc, color='red', linestyle='dashed', linewidth=2, label=f'Observed Acc: {true_acc:.3f}')
    plt.title(f"{model_id} - Permutation Test\n{N_PERMS} Permutations | p-value ≈ {p_val:.3f}")
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{model_id}_perm_hist.png"), dpi=300)
    plt.close()
    return p_val

def block_permute(y, site_ids):
    y_perm = np.copy(y)
    for site in np.unique(site_ids):
        mask = (site_ids == site)
        y_perm[mask] = np.random.permutation(y[mask])
    return y_perm

def run_true_evaluation(X_spd, y, strat_labels, site_ids, scenario, clf_base, param_grid, model_id, out_dir):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    fold_results = []
    site_metrics_list = []
    cm_total = np.zeros((2, 2), dtype=int)
    
    print(f"\n--- Running True Evaluation (5-Fold) for {model_id} ---")
    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_spd, strat_labels), 1):
        fold_start = time.time()  
        
        X_tr, X_te = process_features(
            X_spd[tr_idx], X_spd[te_idx], y[tr_idx], y[te_idx], site_ids[tr_idx], site_ids[te_idx], scenario
        )

        grid = GridSearchCV(clf_base, param_grid, cv=list(inner_cv.split(X_tr, strat_labels[tr_idx])), 
                            scoring="roc_auc", n_jobs=PATHS["n_jobs"], return_train_score=True)
        grid.fit(X_tr, y[tr_idx])
        
        best_model = grid.best_estimator_
        probs = best_model.predict_proba(X_te)[:, 1] if hasattr(best_model, "predict_proba") else best_model.decision_function(X_te)
        preds = best_model.predict(X_te)
    
        cm = confusion_matrix(y[te_idx], preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        cm_total += cm
        
        fold_results.append({
            "Model": model_id, "Fold": fold, 
            "AUC": roc_auc_score(y[te_idx], probs), "ACC": accuracy_score(y[te_idx], preds),
            "F1": f1_score(y[te_idx], preds, zero_division=0), 
            "Sens": tp/(tp+fn) if (tp+fn)>0 else 0, 
            "Spec": tn/(tn+fp) if (tn+fp)>0 else 0,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "Train_AUC": grid.cv_results_['mean_train_score'][grid.best_index_], 
            "Params": str(grid.best_params_)
        })
        
        site_te = site_ids[te_idx]
        y_te_labels = y[te_idx]
        for site in np.unique(site_te):
            mask = (site_te == site)
            if np.sum(mask) > 0:
                y_s = y_te_labels[mask]
                p_s = preds[mask]
                cm_s = confusion_matrix(y_s, p_s, labels=[0, 1])
                tn_s, fp_s, fn_s, tp_s = cm_s.ravel()
                
                site_metrics_list.append({
                    "Model": model_id, "Fold": fold, "Site": site, "N": np.sum(mask), 
                    "ACC": accuracy_score(y_s, p_s),
                    "F1": f1_score(y_s, p_s, zero_division=0),
                    "Sens": tp_s/(tp_s+fn_s) if (tp_s+fn_s)>0 else 0,
                    "Spec": tn_s/(tn_s+fp_s) if (tn_s+fp_s)>0 else 0,
                    "TN": tn_s, "FP": fp_s, "FN": fn_s, "TP": tp_s
                })
                
        fold_time = time.time() - fold_start  
        print(f"      Fold {fold} | ACC: {fold_results[-1]['ACC']:.3f} | AUC: {fold_results[-1]['AUC']:.3f} | Sens: {fold_results[-1]['Sens']:.3f} | Spec: {fold_results[-1]['Spec']:.3f} | Time: {fold_time:.1f}s")
        gc.collect()

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_total, annot=True, fmt='d', cmap='Blues', xticklabels=['Control', 'ASD'], yticklabels=['Control', 'ASD'])
    plt.title(f"{model_id}\nTotal Confusion Matrix (5-CV)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_id}_cm.png"), dpi=300)
    plt.close()

    return fold_results, cm_total, pd.DataFrame(site_metrics_list)

def run_permutation_test(X_spd, y, strat_labels, site_ids, scenario, clf_base, param_grid):
    outer_cv = StratifiedKFold(n_splits=3, shuffle=True)
    inner_cv = StratifiedKFold(n_splits=2, shuffle=True)
    
    perm_acc_list = []
    
    for p in range(N_PERMS):
        perm_start = time.time()
        y_perm = block_permute(y, site_ids)
        strat_perm = np.array([f"{s}_{l}" for s, l in zip(site_ids, y_perm)])
        strat_perm = np.array([s if pd.Series(strat_perm).value_counts()[s] >= 3 else "Other" for s in strat_perm])
        
        fold_accs = []
        for tr_idx, te_idx in outer_cv.split(X_spd, strat_perm):
            X_tr, X_te = process_features(
                X_spd[tr_idx], X_spd[te_idx], y_perm[tr_idx], y_perm[te_idx], site_ids[tr_idx], site_ids[te_idx], scenario
            )
            
            grid = GridSearchCV(clf_base, param_grid, cv=inner_cv, scoring="accuracy", n_jobs=PATHS["n_jobs"])
            grid.fit(X_tr, y_perm[tr_idx])
            fold_accs.append(accuracy_score(y_perm[te_idx], grid.predict(X_te)))
            
        perm_acc_list.append(np.mean(fold_accs))
        if (p+1) % 10 == 0:
            avg_time = time.time() - perm_start
            print(f"      [Perm {p+1:02d}/{N_PERMS}] Completed. (~{avg_time:.1f}s per iteration)")
            
    return perm_acc_list

def execute_master_study(atlas_name="cc200"):
    print(f"Loading data for {atlas_name.upper()}...")
    with open(PATHS["input"], "rb") as f: df = pickle.load(f).dropna(subset=["ASD", "SITE_ID"])

    X_spd = np.ascontiguousarray(np.stack(df[f"corr_{atlas_name}"].values), dtype=np.float64)
    y, site_ids = df["ASD"].values, df["SITE_ID"].values
    
    df["strat"] = df["SITE_ID"].astype(str) + "_" + df["ASD"].astype(str)
    strat_labels = np.array([s if df["strat"].value_counts()[s] >= 5 else "Other" for s in df["strat"]])

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

    scenarios = ['standard_scaler', 'mean_centering', 'procrustes', 'neuroHarmonize']

    for clf_name, config in classifiers.items():
        for sc in scenarios:
            model_id = f"{clf_name}_{sc}"
            out_dir = os.path.join(PATHS["output"], model_id)
            summary_path = os.path.join(out_dir, f"{model_id}_summary.csv")
            
            if os.path.exists(summary_path):
                print(f"Skipping {model_id} (Already Completed)...")
                continue
                
            os.makedirs(out_dir, exist_ok=True)
            print(f"\n{'='*60}\n🚀 STARTING {model_id}\n{'='*60}")
            
            true_results, cm_total, site_df = run_true_evaluation(
                X_spd, y, strat_labels, site_ids, sc, config["clf"], config["grid"], model_id, out_dir
            )
            
            print(f"\n--- Starting {N_PERMS} Permutations (3-Fold Speed CV) ---")
            null_distribution = run_permutation_test(
                X_spd, y, strat_labels, site_ids, sc, config["clf"], config["grid"]
            )

            fold_df = pd.DataFrame(true_results)
            fold_df.to_csv(os.path.join(out_dir, f"{model_id}_detailed_folds.csv"), index=False)
            

            site_df.to_csv(os.path.join(out_dir, f"{model_id}_site_folds_detail.csv"), index=False)
        
            site_summary = site_df.groupby("Site").agg(
                N_Total=("N", "sum"),
                ACC_Mean=("ACC", "mean"), ACC_Std=("ACC", "std"),
                F1_Mean=("F1", "mean"),
                Sens_Mean=("Sens", "mean"),
                Spec_Mean=("Spec", "mean"),
                TN_Total=("TN", "sum"), FP_Total=("FP", "sum"), 
                FN_Total=("FN", "sum"), TP_Total=("TP", "sum")
            ).reset_index()
            site_summary.to_csv(os.path.join(out_dir, f"{model_id}_site_summary_final.csv"), index=False)
            
            true_acc = np.mean(fold_df["ACC"])
            p_val = save_permutation_plot(true_acc, null_distribution, model_id, out_dir)
            
            summary_dict = {
                "Model": model_id, "P_Value": p_val,
                "AUC_Mean": np.mean(fold_df["AUC"]), "ACC_Mean": true_acc,
                "F1_Mean": np.mean(fold_df["F1"]), "Sens_Mean": np.mean(fold_df["Sens"]), "Spec_Mean": np.mean(fold_df["Spec"]),
                "Train_AUC_Mean": np.mean(fold_df["Train_AUC"]),
                "TN": cm_total[0, 0], "FP": cm_total[0, 1], "FN": cm_total[1, 0], "TP": cm_total[1, 1]
            }
            pd.DataFrame([summary_dict]).to_csv(summary_path, index=False)
            
            print(f" {model_id} Completed. ACC: {true_acc:.3f} | Sens: {summary_dict['Sens_Mean']:.3f} | p-value: {p_val:.3f}")

if __name__ == "__main__":
    execute_master_study()