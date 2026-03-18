import numpy as np
import pandas as pd
import pickle
import os
import gc
import warnings
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from pyriemann.tangentspace import TangentSpace

warnings.filterwarnings("ignore")
np.random.seed(42)

N_PERMS = 50 


def block_permute(y, site_ids):
    y_perm = np.copy(y)
    for site in np.unique(site_ids):
        mask = (site_ids == site)
        y_perm[mask] = np.random.permutation(y[mask])
    return y_perm

def save_confusion_plot(cm, model_id, sens, spec, folder):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Control', 'ASD'], yticklabels=['Control', 'ASD'])
    plt.title(f"{model_id}\nSens {sens:.2f} | Spec {spec:.2f}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{model_id}_cm.png"), dpi=300)
    plt.close()

def save_permutation_plot(true_acc, perm_accs, p_val, model_id, folder):
    plt.figure(figsize=(7, 5))
    plt.hist(perm_accs, bins=10, color='lightgray', edgecolor='black', alpha=0.7, label='Null Distribution')
    plt.axvline(true_acc, color='red', linestyle='dashed', linewidth=2, label=f'Observed Acc: {true_acc:.3f}')
    plt.title(f"{model_id} - Permutation Test\n{N_PERMS} Permutations | p-value ≈ {p_val:.3f}")
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{model_id}_perm_hist.png"), dpi=300)
    plt.close()


def run_true_evaluation(X_spd, y, strat_labels, clf_base, param_grid, dim_red_type, model_id):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    fold_results = []
    cm_total = np.zeros((2, 2), dtype=int)
    
    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_spd, strat_labels), 1):
        X_tr_raw, X_te_raw = X_spd[tr_idx], X_spd[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        
        ts = TangentSpace(metric='riemann')
        X_tr = ts.fit_transform(X_tr_raw)
        X_te = ts.transform(X_te_raw)

        scaler = StandardScaler(with_mean=True, with_std=False)
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        if dim_red_type == 'pca':
            dim_step = ("pca", PCA(random_state=42))
        elif dim_red_type == 'anova':
            dim_step = ("anova", SelectKBest(score_func=f_classif))
            
        pipe = Pipeline([dim_step, ("clf", clf_base)])
        grid = GridSearchCV(pipe, param_grid, 
                            cv=list(inner_cv.split(X_tr, strat_labels[tr_idx])), 
                            scoring="roc_auc", n_jobs=-1)
        grid.fit(X_tr, y_tr)
        
        best_model = grid.best_estimator_
        
        if hasattr(best_model.named_steps['clf'], "predict_proba"):
            probs = best_model.predict_proba(X_te)[:, 1]
        else:
            probs = best_model.decision_function(X_te)
            
        preds = best_model.predict(X_te)
        
        cm = confusion_matrix(y_te, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        cm_total += cm
        
        fold_results.append({
            "Model": model_id,
            "Fold": fold,
            "AUC": roc_auc_score(y_te, probs),
            "ACC": accuracy_score(y_te, preds),
            "F1": f1_score(y_te, preds, zero_division=0),
            "Sens": tp/(tp+fn) if (tp+fn)>0 else 0,
            "Spec": tn/(tn+fp) if (tn+fp)>0 else 0,
            "Params": str(grid.best_params_)
        })
        
        print(f"Test ACC: {fold_results[-1]['ACC']:.3f} | Test AUC: {fold_results[-1]['AUC']:.3f} | Params: {grid.best_params_}")
        gc.collect()

    return fold_results, cm_total

def run_permutation_test(X_spd, y, strat_labels, site_ids, clf_base, param_grid, dim_red_type):
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
            X_tr_raw, X_te_raw = X_spd[tr_idx], X_spd[te_idx]
            y_tr, y_te = y_perm[tr_idx], y_perm[te_idx]
            
            ts = TangentSpace(metric='riemann')
            X_tr = ts.fit_transform(X_tr_raw)
            X_te = ts.transform(X_te_raw)

            scaler = StandardScaler(with_mean=True, with_std=False)
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            if dim_red_type == 'pca':
                dim_step = ("pca", PCA(random_state=np.random.randint(10000)))
            elif dim_red_type == 'anova':
                dim_step = ("anova", SelectKBest(score_func=f_classif))
                
            pipe = Pipeline([dim_step, ("clf", clf_base)])

            grid = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="accuracy", n_jobs=-1)
            grid.fit(X_tr, y_tr)
            
            fold_accs.append(accuracy_score(y_te, grid.predict(X_te)))
            
        perm_acc_list.append(np.mean(fold_accs))
        if (p+1) % 10 == 0:
            avg_time = time.time() - perm_start
            print(f"      [Perm {p+1:02d}/{N_PERMS}] Completed. (~{avg_time:.1f}s per iteration)")
            
    return perm_acc_list



def run_dim_reduction_study(atlas_name="cc200"):
    out_dir = "Dimensionality_Reduction_Results"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"📂 Loading data for {atlas_name.upper()}...")
    with open("finalized_corr_shrinkage.pkl", "rb") as f:
        df = pickle.load(f).dropna(subset=["ASD", "SITE_ID"])

    X_spd = np.ascontiguousarray(np.stack(df[f"corr_{atlas_name}"].values), dtype=np.float64)
    y = df["ASD"].values
    site_ids = df["SITE_ID"].values
    
    df["strat"] = df["SITE_ID"].astype(str) + "_" + df["ASD"].astype(str)
    counts = df["strat"].value_counts()
    strat_labels = np.array([s if counts[s] >= 5 else "Other" for s in df["strat"]])


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

    dim_strategies = {
        "PCA": {
            "type": "pca",
            "grid": {
                "pca__n_components": [0.70, 0.80, 0.90]
            }
        },
        "ANOVA": {
            "type": "anova",
            "grid": {
                "anova__k": [1000, 2000, 3000, 4000, 5000]
            }
        }
    }

    for clf_name, clf_config in classifiers.items():
        for dim_name, strategy in dim_strategies.items():
            model_id = f"{clf_name}_{dim_name}"

            combined_grid = strategy["grid"].copy()
            for param, values in clf_config["grid"].items():
                combined_grid[f"clf__{param}"] = values
            
            print(f"\n{'='*50}\n STARTING SCENARIO: {model_id}\n{'='*50}")
            

            fold_results, cm_total = run_true_evaluation(
                X_spd, y, strat_labels, clf_config["clf"], combined_grid, strategy["type"], model_id
            )
            
            print(f"\n--- Starting {N_PERMS} Permutations ---")
            perm_acc_list = run_permutation_test(
                X_spd, y, strat_labels, site_ids, clf_config["clf"], combined_grid, strategy["type"]
            )

            fold_df = pd.DataFrame(fold_results)
            true_acc_mean = np.mean(fold_df["ACC"])
            p_val = (np.sum(np.array(perm_acc_list) >= true_acc_mean) + 1) / (N_PERMS + 1)
            
            sens_mean = np.mean(fold_df["Sens"])
            spec_mean = np.mean(fold_df["Spec"])
            fold_df.to_csv(os.path.join(out_dir, f"{model_id}_detailed_folds.csv"), index=False)
            
            summary_dict = {
                "Model": model_id,
                "P_Value_Perm": round(p_val, 4),
                "AUC_Mean": np.mean(fold_df["AUC"]), "AUC_STD": np.std(fold_df["AUC"]),
                "ACC_Mean": true_acc_mean, "ACC_STD": np.std(fold_df["ACC"]),
                "F1_Mean": np.mean(fold_df["F1"]), "F1_STD": np.std(fold_df["F1"]),
                "Sens_Mean": sens_mean, "Sens_STD": np.std(fold_df["Sens"]),
                "Spec_Mean": spec_mean, "Spec_STD": np.std(fold_df["Spec"]),
                "TN": cm_total[0, 0], "FP": cm_total[0, 1],
                "FN": cm_total[1, 0], "TP": cm_total[1, 1]
            }
            pd.DataFrame([summary_dict]).to_csv(os.path.join(out_dir, f"{model_id}_summary.csv"), index=False)
            
            save_confusion_plot(cm_total, model_id, sens_mean, spec_mean, out_dir)
            save_permutation_plot(true_acc_mean, perm_acc_list, p_val, model_id, out_dir)

            print(f"{model_id} Completed. ACC: {true_acc_mean:.3f} | p-value: {p_val:.3f}")

if __name__ == "__main__":
    run_dim_reduction_study()