import os
import pandas as pd
import numpy as np
import pickle
import re
from tqdm import tqdm
from sklearn.covariance import OAS

def normalize_id(raw_id):
    try:
        return str(int(raw_id))
    except (ValueError, TypeError):
        return None

def process_abide_oas_geometric(data_dir="atlases", pheno_path="pheno_file.csv", output_file="finalized_corr_shrinkage.pkl"):

    atlas_config = {
        'cc200': {'rois': 200},
        'aal': {'rois': 116},
        'dos160': {'rois': 161} 
    }
    if not os.path.exists(pheno_path):
        raise FileNotFoundError(f"Missing phenotypic file: {pheno_path}")

    print("Loading phenotypic labels...")
    df_labels = pd.read_csv(pheno_path)
    df_labels['ASD'] = df_labels['DX_GROUP'].map({1: 1, 2: 0})
    df_labels['norm_id'] = df_labels['SUB_ID'].astype(str).apply(normalize_id)
    df_labels["SEX_01"] = df_labels['SEX'].map({1: 1, 2: 0})
    df_labels['SITE_ID'] = df_labels['SITE_ID'].astype(str).str.replace(r'_\d+$', '', regex=True)
    
    meta_lookup = df_labels.set_index('norm_id')[['ASD', 'SITE_ID', "AGE_AT_SCAN", "SEX_01"]].to_dict('index')
    subject_map = {}
    
    oas_estimator = OAS(store_precision=False, assume_centered=True)

    for atlas, config in atlas_config.items():
        atlas_path = os.path.join(data_dir, atlas)
        if not os.path.exists(atlas_path):
            continue

        print(f"Processing {atlas} (ROIs: {config['rois']}) for Geometric Pipeline...")
        files = [f for f in os.listdir(atlas_path) if not f.startswith('.')]
        
        for subject_file in tqdm(files, desc=f"Extracting {atlas}"):
            parts = subject_file.split("_")
            norm_id = None
            try:
                if len(parts) >= 3:
                    norm_id = normalize_id(parts[-3][2:])
                if not norm_id: 
                    match = re.search(r'5\d{4,6}', subject_file)
                    if match: norm_id = normalize_id(match.group())
            except: continue

            if norm_id not in meta_lookup: continue 

            file_path = os.path.join(atlas_path, subject_file)
            try:
                ts_data = pd.read_csv(file_path, sep="\t")
                
                if ts_data.shape[1] != config['rois']:
                    if ts_data.iloc[:, 0].name.lower() in ['time', '#time']:
                        ts_data = ts_data.iloc[:, 1:]

                if ts_data.shape[1] != config['rois']:
                    continue 

                X = ts_data.values
                X = X - np.mean(X, axis=0, keepdims=True)
                var = np.var(X, axis=0)
                var_floor = 1e-6
                var[var < var_floor] = var_floor
                std = np.sqrt(var)

                X = X / std  

                oas_estimator.fit(X)
                cov_matrix = oas_estimator.covariance_

                d = np.sqrt(np.diag(cov_matrix))
                corr_matrix = cov_matrix / np.outer(d, d)

                if norm_id not in subject_map:
                    subject_map[norm_id] = {
                        'id': norm_id,
                        'ASD': meta_lookup[norm_id]['ASD'],
                        'SITE_ID': meta_lookup[norm_id]['SITE_ID'],
                        "AGE_AT_SCAN": meta_lookup[norm_id]['AGE_AT_SCAN'],
                        "SEX_01": meta_lookup[norm_id]['SEX_01']
                    }
                
                subject_map[norm_id][f'corr_{atlas}'] = corr_matrix

            except Exception as e:
                continue

    df_final = pd.DataFrame(list(subject_map.values()))
    required_cols = [f'corr_{a}' for a in atlas_config.keys()]

    df_final = df_final.dropna(subset=[c for c in required_cols if c in df_final.columns])
    
    print(f"\n--- Final Dataset Statistics ---")
    print(f"Total Subjects: {len(df_final)}")
    if not df_final.empty:
        print(f"Feature Shapes (OAS SPD Matrices):")
        for col in required_cols:
            if col in df_final.columns:
                print(f"  {col}: {df_final[col].iloc[0].shape}")

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(df_final, f)
    
    return df_final

if __name__ == "__main__":
    df = process_abide_oas_geometric()