import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import optuna
from optuna.samplers import TPESampler
import warnings
import json
import sys
warnings.filterwarnings('ignore')

# Suppress Optuna logs for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# Neural Network Classes (imported concept from 02_hyperparameter_optimization.py)
# ============================================================
class FlexibleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(1)

def train_nn_optuna(X_train, y_train, X_val, y_val, params):
    """Train neural network with given hyperparameters"""
    model = FlexibleNN(
        input_dim=X_train.shape[1],
        hidden_dim1=params['hidden_dim1'],
        hidden_dim2=params['hidden_dim2'],
        dropout_rate=params['dropout_rate']
    )
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    loss_fn = nn.BCEWithLogitsLoss()
    
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)
    
    best_loss = np.inf
    patience = 10
    ctr = 0
    
    for epoch in range(100):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        
        model.eval()
        losses = []
        all_preds = []
        all_true = []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                losses.append(loss_fn(logits, yb).item())
                probs = torch.sigmoid(logits).numpy()
                all_preds.extend(probs)
                all_true.extend(yb.numpy())
        
        val_loss = np.mean(losses)
        if val_loss < best_loss:
            best_loss = val_loss
            best_auc = roc_auc_score(all_true, all_preds)
            best_state = model.state_dict()
            ctr = 0
        else:
            ctr += 1
            if ctr >= patience:
                break
    
    return best_auc

def train_nn_final(X_train, y_train, X_val, y_val, params):
    """Train neural network and return the model"""
    model = FlexibleNN(
        input_dim=X_train.shape[1],
        hidden_dim1=params['hidden_dim1'],
        hidden_dim2=params['hidden_dim2'],
        dropout_rate=params['dropout_rate']
    )
    
    opt = optim.Adam(
        model.parameters(), 
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    loss_fn = nn.BCEWithLogitsLoss()
    
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)
    
    best_loss = np.inf
    patience = 10
    ctr = 0
    
    for epoch in range(100):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                losses.append(loss_fn(model(xb), yb).item())
        
        val_loss = np.mean(losses)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            ctr = 0
        else:
            ctr += 1
            if ctr >= patience:
                break
    
    model.load_state_dict(best_state)
    return model

def predict_nn(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).numpy()
    return probs

# ============================================================
# Load and Balance Dataset
# ============================================================
print("="*60)
print("NESTED 5-FOLD CROSS-VALIDATION")
print("="*60)

ml_df = pd.read_pickle('feature_matrix.pkl')
print(f"\nLoaded feature matrix with shape: {ml_df.shape}")
print(f"Total - Controls: {(ml_df['ASD'] == 0).sum()}, Autism: {(ml_df['ASD'] == 1).sum()}")

# ============================================================
# Balance the full dataset (site-stratified)
# ============================================================
print("\n" + "="*60)
print("BALANCING FULL DATASET")
print("="*60)

ml_df_controls = ml_df[ml_df['ASD'] == 0].reset_index(drop=True)
ml_df_autism = ml_df[ml_df['ASD'] == 1].reset_index(drop=True)

n_controls_total = len(ml_df_controls)
n_autism_total = len(ml_df_autism)
n_to_remove = n_controls_total - n_autism_total

print(f"\nRemoving {n_to_remove} control samples to balance dataset")

# Remove proportionally from each site
controls_by_site = ml_df_controls.groupby('SITE_ID').size()
indices_to_remove = []

for site_id, count in controls_by_site.items():
    n_remove_from_site = int(np.round(count * n_to_remove / n_controls_total))
    site_control_indices = ml_df_controls[ml_df_controls['SITE_ID'] == site_id].index.tolist()
    
    np.random.seed(42)
    if n_remove_from_site > 0 and n_remove_from_site <= len(site_control_indices):
        remove_from_site = np.random.choice(site_control_indices, size=n_remove_from_site, replace=False)
        indices_to_remove.extend(remove_from_site)

# Adjust if needed
current_removed = len(indices_to_remove)
if current_removed < n_to_remove:
    remaining_controls = [idx for idx in ml_df_controls.index if idx not in indices_to_remove]
    np.random.seed(42)
    additional = np.random.choice(remaining_controls, size=n_to_remove - current_removed, replace=False)
    indices_to_remove.extend(additional)
elif current_removed > n_to_remove:
    indices_to_remove = indices_to_remove[:n_to_remove]

ml_df_controls_balanced = ml_df_controls.drop(indices_to_remove).reset_index(drop=True)
ml_df_balanced = pd.concat([ml_df_controls_balanced, ml_df_autism], ignore_index=True)

print(f"Balanced dataset - Controls: {len(ml_df_controls_balanced)}, Autism: {len(ml_df_autism)}")
print(f"Total samples: {len(ml_df_balanced)}")

# Create stratification variable for CV (site + label)
ml_df_balanced['stratify_var'] = ml_df_balanced['SITE_ID'].astype(str) + '_' + ml_df_balanced['ASD'].astype(str)

# ============================================================
# Setup 5-Fold Outer Cross-Validation
# ============================================================
print("\n" + "="*60)
print("SETTING UP 5-FOLD OUTER CROSS-VALIDATION")
print("="*60)

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Verify the splits
print("\nVerifying fold balance:")
for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(
    ml_df_balanced.drop(columns=['stratify_var']), 
    ml_df_balanced['stratify_var']
), 1):
    train_df = ml_df_balanced.iloc[train_idx]
    test_df = ml_df_balanced.iloc[test_idx]
    
    train_controls = (train_df['ASD'] == 0).sum()
    train_autism = (train_df['ASD'] == 1).sum()
    test_controls = (test_df['ASD'] == 0).sum()
    test_autism = (test_df['ASD'] == 1).sum()
    
    train_sites = train_df['SITE_ID'].nunique()
    test_sites = test_df['SITE_ID'].nunique()
    
    print(f"\nFold {fold_idx}:")
    print(f"  Training (80%): {len(train_df)} samples")
    print(f"    Controls: {train_controls}, Autism: {train_autism}")
    print(f"    Sites: {train_sites}")
    print(f"  Testing (20%): {len(test_df)} samples")
    print(f"    Controls: {test_controls}, Autism: {test_autism}")
    print(f"    Sites: {test_sites}")

print("\n" + "="*60)
print("STARTING NESTED CROSS-VALIDATION")
print("="*60)

# Store results for all outer folds
all_results = {
    'Linear SVM': {'accuracy': [], 'auc': [], 'y_true': [], 'y_pred_proba': []},
    'RBF SVM': {'accuracy': [], 'auc': [], 'y_true': [], 'y_pred_proba': []},
    'Random Forest': {'accuracy': [], 'auc': [], 'y_true': [], 'y_pred_proba': []},
    'Neural Network': {'accuracy': [], 'auc': [], 'y_true': [], 'y_pred_proba': []}
}

# Store best parameters from each fold
all_best_params = []

# ============================================================
# Run Nested CV - Each Outer Fold
# ============================================================
n_trials = 30  # Number of Optuna trials per model

for outer_fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(
    ml_df_balanced.drop(columns=['stratify_var']), 
    ml_df_balanced['stratify_var']
), 1):
    
    print(f"\n{'='*60}")
    print(f"OUTER FOLD {outer_fold_idx}/5")
    print(f"{'='*60}")
    
    # Split data
    train_df = ml_df_balanced.iloc[train_idx].reset_index(drop=True)
    test_df = ml_df_balanced.iloc[test_idx].reset_index(drop=True)
    
    # Extract features
    X_train_outer = train_df.drop(columns=["id", "SITE_ID", "ASD", "stratify_var"]).values
    y_train_outer = train_df["ASD"].values
    stratify_labels_outer = train_df['SITE_ID'].astype(str) + '_' + train_df['ASD'].astype(str)
    
    X_test_outer = test_df.drop(columns=["id", "SITE_ID", "ASD", "stratify_var"]).values
    y_test_outer = test_df["ASD"].values
    
    print(f"\nTrain: {X_train_outer.shape}, Test: {X_test_outer.shape}")
    
    # Calculate max PCA components for this fold
    min_fold_size = int(X_train_outer.shape[0] * 0.80)  # 5-fold inner CV
    max_pca_components = min(min_fold_size, X_train_outer.shape[1])
    print(f"Max PCA components for this fold: {max_pca_components}")
    
    # ============================================================
    # Hyperparameter Optimization with Optuna (Inner CV)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER OPTIMIZATION (Inner 5-Fold CV)")
    print(f"{'='*60}")
    
    best_params_fold = {}
    
    # Define objective functions for this fold
    def objective_rbf_svm(trial):
        C = trial.suggest_float('C', 1e-3, 1e2, log=True)
        gamma = trial.suggest_float('gamma', 1e-5, 1e-1, log=True)
        n_components = trial.suggest_int('n_components', 10, max_pca_components - 5, step=10)
        
        model = SVC(kernel="rbf", C=C, gamma=gamma, random_state=42)
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        auc_scores = []
        for train_idx_inner, val_idx_inner in inner_cv.split(X_train_outer, stratify_labels_outer):
            X_train_inner, X_val_inner = X_train_outer[train_idx_inner], X_train_outer[val_idx_inner]
            y_train_inner, y_val_inner = y_train_outer[train_idx_inner], y_train_outer[val_idx_inner]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_inner)
            X_val_scaled = scaler.transform(X_val_inner)
            
            pca = PCA(n_components=n_components, random_state=42)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_val_pca = pca.transform(X_val_scaled)
            
            model.fit(X_train_pca, y_train_inner)
            y_score = model.decision_function(X_val_pca)
            
            try:
                score = roc_auc_score(y_val_inner, y_score)
                auc_scores.append(score)
            except:
                auc_scores.append(0.5)
        
        return np.mean(auc_scores)
    
    def objective_random_forest(trial):
        n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)
        max_depth = trial.suggest_int('max_depth', 5, 50)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        n_components = trial.suggest_int('n_components', 10, max_pca_components - 5, step=10)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=42
        )
        
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        auc_scores = []
        for train_idx_inner, val_idx_inner in inner_cv.split(X_train_outer, stratify_labels_outer):
            X_train_inner, X_val_inner = X_train_outer[train_idx_inner], X_train_outer[val_idx_inner]
            y_train_inner, y_val_inner = y_train_outer[train_idx_inner], y_train_outer[val_idx_inner]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_inner)
            X_val_scaled = scaler.transform(X_val_inner)
            
            pca = PCA(n_components=n_components, random_state=42)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_val_pca = pca.transform(X_val_scaled)
            
            model.fit(X_train_pca, y_train_inner)
            y_proba = model.predict_proba(X_val_pca)[:, 1]
            
            try:
                score = roc_auc_score(y_val_inner, y_proba)
                auc_scores.append(score)
            except:
                auc_scores.append(0.5)
        
        return np.mean(auc_scores)
    
    def objective_neural_network(trial):
        params = {
            'hidden_dim1': trial.suggest_int('hidden_dim1', 64, 512, step=64),
            'hidden_dim2': trial.suggest_int('hidden_dim2', 32, 256, step=32),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.7),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'n_components': trial.suggest_int('n_components', 10, max_pca_components - 5, step=10)
        }
        
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = []
        
        for train_idx_inner, val_idx_inner in inner_cv.split(X_train_outer, stratify_labels_outer):
            X_train_inner, X_val_inner = X_train_outer[train_idx_inner], X_train_outer[val_idx_inner]
            y_train_inner, y_val_inner = y_train_outer[train_idx_inner], y_train_outer[val_idx_inner]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_inner)
            X_val_scaled = scaler.transform(X_val_inner)
            
            pca = PCA(n_components=params['n_components'], random_state=42)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_val_pca = pca.transform(X_val_scaled)
            
            try:
                auc = train_nn_optuna(X_train_pca, y_train_inner, X_val_pca, y_val_inner, params)
                auc_scores.append(auc)
            except:
                auc_scores.append(0.5)
        
        return np.mean(auc_scores)
    
    # Linear SVM - default parameters
    print("\nLinear SVM - Using default parameters (C=1.0, no PCA)")
    best_params_fold['Linear SVM'] = {'C': 1.0}
    
    # RBF SVM
    print(f"\nOptimizing RBF SVM ({n_trials} trials)...")
    study_rbf = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_rbf.optimize(objective_rbf_svm, n_trials=n_trials, show_progress_bar=False)
    best_params_fold['RBF SVM'] = study_rbf.best_params
    print(f"  Best AUC: {study_rbf.best_value:.4f}")
    
    # Random Forest
    print(f"\nOptimizing Random Forest ({n_trials} trials)...")
    study_rf = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_rf.optimize(objective_random_forest, n_trials=n_trials, show_progress_bar=False)
    best_params_fold['Random Forest'] = study_rf.best_params
    print(f"  Best AUC: {study_rf.best_value:.4f}")
    
    # Neural Network
    print(f"\nOptimizing Neural Network ({n_trials} trials)...")
    study_nn = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_nn.optimize(objective_neural_network, n_trials=n_trials, show_progress_bar=False)
    best_params_fold['Neural Network'] = study_nn.best_params
    print(f"  Best AUC: {study_nn.best_value:.4f}")
    
    all_best_params.append(best_params_fold)
    
    # ============================================================
    # Train Final Models and Evaluate on Outer Test Fold
    # ============================================================
    print(f"\n{'='*60}")
    print(f"TRAINING AND EVALUATING ON OUTER TEST FOLD")
    print(f"{'='*60}")
    
    # Standardize data for this outer fold
    scaler_outer = StandardScaler()
    X_train_outer_scaled = scaler_outer.fit_transform(X_train_outer)
    X_test_outer_scaled = scaler_outer.transform(X_test_outer)
    
    # Linear SVM
    print("\nLinear SVM...")
    model_linear = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    model_linear.fit(X_train_outer_scaled, y_train_outer)
    y_pred_proba = model_linear.predict_proba(X_test_outer_scaled)[:, 1]
    y_pred = model_linear.predict(X_test_outer_scaled)
    
    acc = accuracy_score(y_test_outer, y_pred)
    fpr, tpr, _ = roc_curve(y_test_outer, y_pred_proba)
    auc_score = auc(fpr, tpr)
    
    all_results['Linear SVM']['accuracy'].append(acc)
    all_results['Linear SVM']['auc'].append(auc_score)
    all_results['Linear SVM']['y_true'].extend(y_test_outer.tolist())
    all_results['Linear SVM']['y_pred_proba'].extend(y_pred_proba.tolist())
    print(f"  Accuracy: {acc:.4f}, AUC: {auc_score:.4f}")
    
    # RBF SVM
    print("\nRBF SVM...")
    n_comp = best_params_fold['RBF SVM']['n_components']
    pca_rbf = PCA(n_components=n_comp, random_state=42)
    X_train_pca = pca_rbf.fit_transform(X_train_outer_scaled)
    X_test_pca = pca_rbf.transform(X_test_outer_scaled)
    
    model_rbf = SVC(
        kernel='rbf',
        C=best_params_fold['RBF SVM']['C'],
        gamma=best_params_fold['RBF SVM']['gamma'],
        probability=True,
        random_state=42
    )
    model_rbf.fit(X_train_pca, y_train_outer)
    y_pred_proba = model_rbf.predict_proba(X_test_pca)[:, 1]
    y_pred = model_rbf.predict(X_test_pca)
    
    acc = accuracy_score(y_test_outer, y_pred)
    fpr, tpr, _ = roc_curve(y_test_outer, y_pred_proba)
    auc_score = auc(fpr, tpr)
    
    all_results['RBF SVM']['accuracy'].append(acc)
    all_results['RBF SVM']['auc'].append(auc_score)
    all_results['RBF SVM']['y_true'].extend(y_test_outer.tolist())
    all_results['RBF SVM']['y_pred_proba'].extend(y_pred_proba.tolist())
    print(f"  Accuracy: {acc:.4f}, AUC: {auc_score:.4f}")
    
    # Random Forest
    print("\nRandom Forest...")
    n_comp = best_params_fold['Random Forest'].pop('n_components')
    pca_rf = PCA(n_components=n_comp, random_state=42)
    X_train_pca = pca_rf.fit_transform(X_train_outer_scaled)
    X_test_pca = pca_rf.transform(X_test_outer_scaled)
    
    model_rf = RandomForestClassifier(
        **best_params_fold['Random Forest'],
        n_jobs=-1,
        random_state=42
    )
    model_rf.fit(X_train_pca, y_train_outer)
    y_pred_proba = model_rf.predict_proba(X_test_pca)[:, 1]
    y_pred = model_rf.predict(X_test_pca)
    
    acc = accuracy_score(y_test_outer, y_pred)
    fpr, tpr, _ = roc_curve(y_test_outer, y_pred_proba)
    auc_score = auc(fpr, tpr)
    
    all_results['Random Forest']['accuracy'].append(acc)
    all_results['Random Forest']['auc'].append(auc_score)
    all_results['Random Forest']['y_true'].extend(y_test_outer.tolist())
    all_results['Random Forest']['y_pred_proba'].extend(y_pred_proba.tolist())
    print(f"  Accuracy: {acc:.4f}, AUC: {auc_score:.4f}")
    
    # Neural Network
    print("\nNeural Network...")
    nn_params = best_params_fold['Neural Network'].copy()
    n_comp = nn_params.pop('n_components')
    pca_nn = PCA(n_components=n_comp, random_state=42)
    X_train_pca = pca_nn.fit_transform(X_train_outer_scaled)
    X_test_pca = pca_nn.transform(X_test_outer_scaled)
    
    # Split training data for NN validation
    from sklearn.model_selection import train_test_split
    X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
        X_train_pca, y_train_outer, test_size=0.2, stratify=y_train_outer, random_state=42
    )
    
    model_nn = train_nn_final(X_train_nn, y_train_nn, X_val_nn, y_val_nn, nn_params)
    y_pred_proba = predict_nn(model_nn, X_test_pca)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    acc = accuracy_score(y_test_outer, y_pred)
    fpr, tpr, _ = roc_curve(y_test_outer, y_pred_proba)
    auc_score = auc(fpr, tpr)
    
    all_results['Neural Network']['accuracy'].append(acc)
    all_results['Neural Network']['auc'].append(auc_score)
    all_results['Neural Network']['y_true'].extend(y_test_outer.tolist())
    all_results['Neural Network']['y_pred_proba'].extend(y_pred_proba.tolist())
    print(f"  Accuracy: {acc:.4f}, AUC: {auc_score:.4f}")

# ============================================================
# Final Results Summary
# ============================================================
print(f"\n{'='*60}")
print("NESTED CV RESULTS SUMMARY (5 OUTER FOLDS)")
print(f"{'='*60}")

summary_data = []
for model_name in all_results.keys():
    acc_mean = np.mean(all_results[model_name]['accuracy'])
    acc_std = np.std(all_results[model_name]['accuracy'])
    acc_se = acc_std / np.sqrt(5)  # Standard error
    
    auc_mean = np.mean(all_results[model_name]['auc'])
    auc_std = np.std(all_results[model_name]['auc'])
    auc_se = auc_std / np.sqrt(5)  # Standard error
    
    summary_data.append({
        'Model': model_name,
        'Accuracy_Mean': acc_mean,
        'Accuracy_Std': acc_std,
        'Accuracy_SE': acc_se,
        'AUC_Mean': auc_mean,
        'AUC_Std': auc_std,
        'AUC_SE': auc_se
    })
    
    print(f"\n{model_name}:")
    print(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f} (SE: {acc_se:.4f})")
    print(f"  AUC: {auc_mean:.4f} ± {auc_std:.4f} (SE: {auc_se:.4f})")
    print(f"  Individual folds:")
    for i, (acc, auc_val) in enumerate(zip(all_results[model_name]['accuracy'], 
                                             all_results[model_name]['auc']), 1):
        print(f"    Fold {i}: Acc={acc:.4f}, AUC={auc_val:.4f}")

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('nested_cv_results.csv', index=False)
print(f"\nResults saved to 'nested_cv_results.csv'")

# Save all best parameters
with open('nested_cv_best_params.json', 'w') as f:
    json.dump(all_best_params, f, indent=4)
print("Best parameters for each fold saved to 'nested_cv_best_params.json'")

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

model_names = list(all_results.keys())
x_pos = np.arange(len(model_names))

acc_means = [np.mean(all_results[m]['accuracy']) for m in model_names]
acc_ses = [np.std(all_results[m]['accuracy']) / np.sqrt(5) for m in model_names]
auc_means = [np.mean(all_results[m]['auc']) for m in model_names]
auc_ses = [np.std(all_results[m]['auc']) / np.sqrt(5) for m in model_names]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Accuracy plot with SE
axes[0].bar(x_pos, acc_means, yerr=acc_ses, capsize=5, alpha=0.7, color=colors)
axes[0].set_xlabel('Model', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[0].set_title('Model Accuracy (Nested 5-Fold CV)\nError bars: Standard Error', 
                  fontsize=14, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(model_names, rotation=45, ha='right')
axes[0].set_ylim([0, 1])
axes[0].grid(axis='y', alpha=0.3)
axes[0].axhline(y=0.5, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Chance')
axes[0].legend()

for i, (mean, se) in enumerate(zip(acc_means, acc_ses)):
    axes[0].text(i, mean + se + 0.02, f'{mean:.3f}', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

# AUC plot with SE
axes[1].bar(x_pos, auc_means, yerr=auc_ses, capsize=5, alpha=0.7, color=colors)
axes[1].set_xlabel('Model', fontsize=12, fontweight='bold')
axes[1].set_ylabel('AUC', fontsize=12, fontweight='bold')
axes[1].set_title('Model AUC (Nested 5-Fold CV)\nError bars: Standard Error', 
                  fontsize=14, fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(model_names, rotation=45, ha='right')
axes[1].set_ylim([0, 1])
axes[1].grid(axis='y', alpha=0.3)
axes[1].axhline(y=0.5, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Chance')
axes[1].legend()

for i, (mean, se) in enumerate(zip(auc_means, auc_ses)):
    axes[1].text(i, mean + se + 0.02, f'{mean:.3f}', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('nested_cv_results.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# ROC Curves - Aggregated across all folds
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))

for idx, model_name in enumerate(model_names):
    y_true = np.array(all_results[model_name]['y_true'])
    y_pred_proba = np.array(all_results[model_name]['y_pred_proba'])
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color=colors[idx], lw=2, 
            label=f'{model_name} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance (AUC = 0.500)')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves - Nested 5-Fold CV\n(Aggregated across all folds)', 
             fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('nested_cv_roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("NESTED CV COMPLETE!")
print("="*60)
print("\nGenerated files:")
print("  - nested_cv_results.csv")
print("  - nested_cv_best_params.json")
print("  - nested_cv_results.png")
print("  - nested_cv_roc_curves.png")