import numpy as np
import pandas as pd

# ============================================================
# Load data
# ============================================================
print("="*60)
print("FEATURE EXTRACTION")
print("="*60)

features_df = pd.read_pickle(
    "/Users/eliezerzucker/Documents/MachineLearning/asd_prediction/abide_raw_pearson.pkl"
)

# ============================================================
# Helper: upper triangle extraction
# ============================================================
def upper_triangle_features(corr_mat, k=1):
    idx = np.triu_indices_from(corr_mat, k=k)
    return corr_mat[idx]

# ============================================================
# Build feature matrix
# ============================================================
print("\nExtracting features from correlation matrices...")

feature_rows = []
feature_names = None
for _, row in features_df.iterrows():
    corr_mat = row["corr_cc200"]
    feats = upper_triangle_features(corr_mat, k=1)
    feature_rows.append(feats)
    if feature_names is None:
        n = corr_mat.shape[0]
        idx = np.triu_indices(n, k=1)
        feature_names = [f"conn_{i}_{j}" for i, j in zip(*idx)]

X_df = pd.DataFrame(feature_rows, columns=feature_names)
ml_df = pd.concat(
    [features_df[["id", "SITE_ID", "ASD"]], X_df],
    axis=1
)

# ============================================================
# Save feature matrix
# ============================================================
output_file = 'feature_matrix.pkl'
ml_df.to_pickle(output_file)

print(f"\nFeature matrix shape: {ml_df.shape}")
print(f"Number of samples: {len(ml_df)}")
print(f"Number of features: {len(feature_names)}")
print(f"Controls: {(ml_df['ASD'] == 0).sum()}")
print(f"Autism: {(ml_df['ASD'] == 1).sum()}")
print(f"\nFeature matrix saved to '{output_file}'")
print("="*60)