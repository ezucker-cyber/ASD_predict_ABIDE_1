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
import json
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
N_PERMUTATIONS = 50  # <<< CHANGE THIS to run more permutations (e.g., 1000)
N_FOLDS = 5
RANDOM_SEED = 42

print("="*60)
print("PERMUTATION TEST FOR MODEL SIGNIFICANCE")
print("="*60)
print(f"\nConfiguration:")
print(f"  Number of permutations: {N_PERMUTATIONS}")
print(f"  Number of CV folds: {N_FOLDS}")
print(f"  Random seed: {RANDOM_SEED}")

# ============================================================
# Neural Network Classes (same as nested CV)
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
# Load Optimized Hyperparameters
# ============================================================
print("\n" + "="*60)
print("LOADING OPTIMIZED HYPERPARAMETERS")
print("="*60)

with open('nested_cv_best_params.json', 'r') as f:
    all_best_params = json.load(f)

# Average the hyperparameters across folds (use most common values)
# For simplicity, we'll use the parameters from Fold 1
best_params = all_best_params[0]

print("\nUsing hyperparameters from Fold 1:")
for model_name, params in best_params.items():
    print(f"\n{model_name}:")
    for param, value in params.items():
        print(f"  {param}: {value}")

# ============================================================
# Load and Balance Dataset
# ============================================================
print("\n" + "="*60)
print("LOADING AND BALANCING DATASET")
print("="*60)

ml_df = pd.read_pickle('feature_matrix.pkl')
print(f"\nLoaded feature matrix: {ml_df.shape}")

# Balance dataset (same as nested CV)
ml_df_controls = ml_df[ml_df['ASD'] == 0].reset_index(drop=True)
ml_df_autism = ml_df[ml_df['ASD'] == 1].reset_index(drop=True)

n_controls_total = len(ml_df_controls)
n_autism_total = len(ml_df_autism)
n_to_remove = n_controls_total - n_autism_total

print(f"Removing {n_to_remove} control samples to balance dataset")

# Remove proportionally from each site
controls_by_site = ml_df_controls.groupby('SITE_ID').size()
indices_to_remove = []

for site_id, count in controls_by_site.items():
    n_remove_from_site = int(np.round(count * n_to_remove / n_controls_total))
    site_control_indices = ml_df_controls[ml_df_controls['SITE_ID'] == site_id].index.tolist()
    
    np.random.seed(RANDOM_SEED)
    if n_remove_from_site > 0 and n_remove_from_site <= len(site_control_indices):
        remove_from_site = np.random.choice(site_control_indices, size=n_remove_from_site, replace=False)
        indices_to_remove.extend(remove_from_site)

# Adjust if needed
current_removed = len(indices_to_remove)
if current_removed < n_to_remove:
    remaining_controls = [idx for idx in ml_df_controls.index if idx not in indices_to_remove]
    np.random.seed(RANDOM_SEED)
    additional = np.random.choice(remaining_controls, size=n_to_remove - current_removed, replace=False)
    indices_to_remove.extend(additional)
elif current_removed > n_to_remove:
    indices_to_remove = indices_to_remove[:n_to_remove]

ml_df_controls_balanced = ml_df_controls.drop(indices_to_remove).reset_index(drop=True)
ml_df_balanced = pd.concat([ml_df_controls_balanced, ml_df_autism], ignore_index=True)

print(f"Balanced dataset - Controls: {len(ml_df_controls_balanced)}, Autism: {len(ml_df_autism)}")
print(f"Total samples: {len(ml_df_balanced)}")

# Create stratification variable
ml_df_balanced['stratify_var'] = ml_df_balanced['SITE_ID'].astype(str) + '_' + ml_df_balanced['ASD'].astype(str)

# ============================================================
# Load Observed Performance (from nested CV)
# ============================================================
print("\n" + "="*60)
print("LOADING OBSERVED PERFORMANCE")
print("="*60)

observed_df = pd.read_csv('nested_cv_results.csv')
observed_performance = {}

for _, row in observed_df.iterrows():
    model_name = row['Model']
    observed_performance[model_name] = {
        'accuracy': row['Accuracy_Mean'],
        'auc': row['AUC_Mean']
    }

print("\nObserved performance from nested CV:")
for model_name, perf in observed_performance.items():
    print(f"  {model_name}: Accuracy={perf['accuracy']:.4f}, AUC={perf['auc']:.4f}")

# ============================================================
# Permutation Test Function
# ============================================================
def run_single_permutation(X_data, y_shuffled, stratify_labels, best_params, perm_idx):
    """Run 5-fold CV with shuffled labels using fixed hyperparameters"""
    
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    results = {
        'Linear SVM': {'accuracy': [], 'auc': []},
        'RBF SVM': {'accuracy': [], 'auc': []},
        'Random Forest': {'accuracy': [], 'auc': []},
        'Neural Network': {'accuracy': [], 'auc': []}
    }
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_data, stratify_labels)):
        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_shuffled[train_idx], y_shuffled[test_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Linear SVM (no PCA)
        model_linear = SVC(kernel='linear', C=best_params['Linear SVM']['C'], 
                          probability=True, random_state=RANDOM_SEED)
        model_linear.fit(X_train_scaled, y_train)
        y_pred_proba = model_linear.predict_proba(X_test_scaled)[:, 1]
        y_pred = model_linear.predict(X_test_scaled)
        
        results['Linear SVM']['accuracy'].append(accuracy_score(y_test, y_pred))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        results['Linear SVM']['auc'].append(auc(fpr, tpr))
        
        # RBF SVM (with PCA)
        n_comp = best_params['RBF SVM'].get('n_components', None)
        
        if n_comp is not None:
            pca_rbf = PCA(n_components=n_comp, random_state=RANDOM_SEED)
            X_train_pca = pca_rbf.fit_transform(X_train_scaled)
            X_test_pca = pca_rbf.transform(X_test_scaled)
        else:
            X_train_pca = X_train_scaled
            X_test_pca = X_test_scaled
        
        model_rbf = SVC(
            kernel='rbf',
            C=best_params['RBF SVM']['C'],
            gamma=best_params['RBF SVM']['gamma'],
            probability=True,
            random_state=RANDOM_SEED
        )
        model_rbf.fit(X_train_pca, y_train)
        y_pred_proba = model_rbf.predict_proba(X_test_pca)[:, 1]
        y_pred = model_rbf.predict(X_test_pca)
        
        results['RBF SVM']['accuracy'].append(accuracy_score(y_test, y_pred))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        results['RBF SVM']['auc'].append(auc(fpr, tpr))
        
        # Random Forest (with PCA)
        rf_params = best_params['Random Forest'].copy()
        n_comp = rf_params.pop('n_components', None)  # Use None as default if key doesn't exist
        
        if n_comp is not None:
            pca_rf = PCA(n_components=n_comp, random_state=RANDOM_SEED)
            X_train_pca = pca_rf.fit_transform(X_train_scaled)
            X_test_pca = pca_rf.transform(X_test_scaled)
        else:
            # No PCA for this model
            X_train_pca = X_train_scaled
            X_test_pca = X_test_scaled
        
        model_rf = RandomForestClassifier(**rf_params, n_jobs=-1, random_state=RANDOM_SEED)
        model_rf.fit(X_train_pca, y_train)
        y_pred_proba = model_rf.predict_proba(X_test_pca)[:, 1]
        y_pred = model_rf.predict(X_test_pca)
        
        results['Random Forest']['accuracy'].append(accuracy_score(y_test, y_pred))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        results['Random Forest']['auc'].append(auc(fpr, tpr))
        
        # Neural Network (with PCA)
        nn_params = best_params['Neural Network'].copy()
        n_comp = nn_params.pop('n_components', None)
        
        if n_comp is not None:
            pca_nn = PCA(n_components=n_comp, random_state=RANDOM_SEED)
            X_train_pca = pca_nn.fit_transform(X_train_scaled)
            X_test_pca = pca_nn.transform(X_test_scaled)
        else:
            X_train_pca = X_train_scaled
            X_test_pca = X_test_scaled
        
        # Split for NN validation
        from sklearn.model_selection import train_test_split
        X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
            X_train_pca, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_SEED
        )
        
        model_nn = train_nn_final(X_train_nn, y_train_nn, X_val_nn, y_val_nn, nn_params)
        y_pred_proba = predict_nn(model_nn, X_test_pca)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        results['Neural Network']['accuracy'].append(accuracy_score(y_test, y_pred))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        results['Neural Network']['auc'].append(auc(fpr, tpr))
    
    # Average across folds
    avg_results = {}
    for model_name in results.keys():
        avg_results[model_name] = {
            'accuracy': np.mean(results[model_name]['accuracy']),
            'auc': np.mean(results[model_name]['auc'])
        }
    
    return avg_results

# ============================================================
# Run Permutation Test
# ============================================================
print("\n" + "="*60)
print(f"RUNNING {N_PERMUTATIONS} PERMUTATIONS")
print("="*60)

# Extract features
X_data = ml_df_balanced.drop(columns=["id", "SITE_ID", "ASD", "stratify_var"]).values
y_true = ml_df_balanced["ASD"].values
stratify_labels = ml_df_balanced['stratify_var'].values

# Store null distribution
null_distribution = {
    'Linear SVM': {'accuracy': [], 'auc': []},
    'RBF SVM': {'accuracy': [], 'auc': []},
    'Random Forest': {'accuracy': [], 'auc': []},
    'Neural Network': {'accuracy': [], 'auc': []}
}

start_time = datetime.now()
print(f"\nStarted at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

for perm_idx in range(N_PERMUTATIONS):
    # Shuffle labels
    np.random.seed(RANDOM_SEED + perm_idx)
    y_shuffled = np.random.permutation(y_true)
    
    # Run CV with shuffled labels
    perm_results = run_single_permutation(X_data, y_shuffled, stratify_labels, best_params, perm_idx)
    
    # Store results
    for model_name in null_distribution.keys():
        null_distribution[model_name]['accuracy'].append(perm_results[model_name]['accuracy'])
        null_distribution[model_name]['auc'].append(perm_results[model_name]['auc'])
    
    # Progress update
    if (perm_idx + 1) % 10 == 0:
        elapsed = datetime.now() - start_time
        avg_time_per_perm = elapsed.total_seconds() / (perm_idx + 1)
        remaining_perms = N_PERMUTATIONS - (perm_idx + 1)
        eta_seconds = avg_time_per_perm * remaining_perms
        eta_minutes = eta_seconds / 60
        
        print(f"  Completed {perm_idx + 1}/{N_PERMUTATIONS} permutations "
              f"(ETA: {eta_minutes:.1f} min)")

end_time = datetime.now()
total_time = end_time - start_time
print(f"\nCompleted at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total time: {total_time}")

# ============================================================
# Calculate P-values
# ============================================================
print("\n" + "="*60)
print("PERMUTATION TEST RESULTS")
print("="*60)

p_values = {}

for model_name in null_distribution.keys():
    obs_acc = observed_performance[model_name]['accuracy']
    obs_auc = observed_performance[model_name]['auc']
    
    null_acc = np.array(null_distribution[model_name]['accuracy'])
    null_auc = np.array(null_distribution[model_name]['auc'])
    
    # P-value = proportion of permutations >= observed
    p_acc = np.sum(null_acc >= obs_acc) / N_PERMUTATIONS
    p_auc = np.sum(null_auc >= obs_auc) / N_PERMUTATIONS
    
    p_values[model_name] = {'accuracy': p_acc, 'auc': p_auc}
    
    print(f"\n{model_name}:")
    print(f"  Observed Accuracy: {obs_acc:.4f}")
    print(f"  Null Mean: {np.mean(null_acc):.4f} ± {np.std(null_acc):.4f}")
    print(f"  p-value (Accuracy): {p_acc:.4f}")
    print(f"  ")
    print(f"  Observed AUC: {obs_auc:.4f}")
    print(f"  Null Mean: {np.mean(null_auc):.4f} ± {np.std(null_auc):.4f}")
    print(f"  p-value (AUC): {p_auc:.4f}")
    
    if p_auc < 0.05:
        print(f"  ✓ SIGNIFICANT (p < 0.05)")
    else:
        print(f"  ✗ Not significant (p ≥ 0.05)")

# ============================================================
# Save Results
# ============================================================
results_summary = []
for model_name in p_values.keys():
    results_summary.append({
        'Model': model_name,
        'Observed_Accuracy': observed_performance[model_name]['accuracy'],
        'Null_Mean_Accuracy': np.mean(null_distribution[model_name]['accuracy']),
        'Null_Std_Accuracy': np.std(null_distribution[model_name]['accuracy']),
        'p_value_Accuracy': p_values[model_name]['accuracy'],
        'Observed_AUC': observed_performance[model_name]['auc'],
        'Null_Mean_AUC': np.mean(null_distribution[model_name]['auc']),
        'Null_Std_AUC': np.std(null_distribution[model_name]['auc']),
        'p_value_AUC': p_values[model_name]['auc']
    })

results_df = pd.DataFrame(results_summary)
results_df.to_csv(f'permutation_test_results_n{N_PERMUTATIONS}.csv', index=False)
print(f"\nResults saved to 'permutation_test_results_n{N_PERMUTATIONS}.csv'")

# Save null distributions
with open(f'permutation_null_distributions_n{N_PERMUTATIONS}.json', 'w') as f:
    # Convert numpy arrays to lists for JSON serialization
    null_dist_serializable = {}
    for model_name in null_distribution.keys():
        null_dist_serializable[model_name] = {
            'accuracy': [float(x) for x in null_distribution[model_name]['accuracy']],
            'auc': [float(x) for x in null_distribution[model_name]['auc']]
        }
    json.dump(null_dist_serializable, f, indent=4)
print(f"Null distributions saved to 'permutation_null_distributions_n{N_PERMUTATIONS}.json'")

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

model_names = list(null_distribution.keys())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for idx, model_name in enumerate(model_names):
    # Accuracy histogram
    ax_acc = axes[0, idx]
    null_acc = null_distribution[model_name]['accuracy']
    obs_acc = observed_performance[model_name]['accuracy']
    p_acc = p_values[model_name]['accuracy']
    
    ax_acc.hist(null_acc, bins=30, alpha=0.7, color=colors[idx], edgecolor='black')
    ax_acc.axvline(obs_acc, color='red', linestyle='--', linewidth=2, 
                   label=f'Observed: {obs_acc:.3f}')
    ax_acc.set_xlabel('Accuracy', fontsize=10, fontweight='bold')
    ax_acc.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax_acc.set_title(f'{model_name}\np = {p_acc:.4f}', 
                     fontsize=11, fontweight='bold')
    ax_acc.legend(fontsize=9)
    ax_acc.grid(alpha=0.3)
    
    # AUC histogram
    ax_auc = axes[1, idx]
    null_auc = null_distribution[model_name]['auc']
    obs_auc = observed_performance[model_name]['auc']
    p_auc = p_values[model_name]['auc']
    
    ax_auc.hist(null_auc, bins=30, alpha=0.7, color=colors[idx], edgecolor='black')
    ax_auc.axvline(obs_auc, color='red', linestyle='--', linewidth=2,
                   label=f'Observed: {obs_auc:.3f}')
    ax_auc.set_xlabel('AUC', fontsize=10, fontweight='bold')
    ax_auc.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax_auc.set_title(f'{model_name}\np = {p_auc:.4f}', 
                     fontsize=11, fontweight='bold')
    ax_auc.legend(fontsize=9)
    ax_auc.grid(alpha=0.3)

fig.suptitle(f'Permutation Test Results (n={N_PERMUTATIONS} permutations)\n'
             f'Red line = Observed performance | Histogram = Null distribution',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'permutation_test_results_n{N_PERMUTATIONS}.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("PERMUTATION TEST COMPLETE!")
print("="*60)
print(f"\nGenerated files:")
print(f"  - permutation_test_results_n{N_PERMUTATIONS}.csv")
print(f"  - permutation_null_distributions_n{N_PERMUTATIONS}.json")
print(f"  - permutation_test_results_n{N_PERMUTATIONS}.png")