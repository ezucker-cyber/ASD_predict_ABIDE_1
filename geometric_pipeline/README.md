# Geometric Machine Learning Pipeline (ABIDE)

This folder contains the scripts for the Riemannian geometry and Multi-Atlas ensemble architecture. 

## Setup & Data Requirements

**Crucually:** 
This pipeline does not include the raw data. You must download the following from [Mina Zeraati and Amirehsan Davoodi](https://github.com/AmirDavoodi/ASD-GraphNet): For this one can follow their instructions. 
Finally: 
* The **`atlases/`** folder (containing the `.1D` ROI time series).
* The **`pheno_file.csv`** (phenotypic metadata).

Place both the folder and the CSV in the same directory as these scripts before running anything.

### Attribution
The logic for directory navigation, subject ID normalization, and linking the `.1D` atlas files to the phenotypic metadata is built on and partially taken from the implementation in **ASD-GraphNet** by [Mina Zeraati and Amirehsan Davoodi]. We use their framework for data ingestion before processing the connectivity matrices through our geometric framework.

For all the specific pre-processing steps used by these authors, see their publication: 
Zeraati, M., & Davoodi, A. (2025). ASD-GraphNet: A novel graph learning         approach for Autism Spectrum Disorder diagnosis using fMRI data. Computers in Biology and Medicine, 196(Pt B), 110723. https://doi.org/10.1016/j.compbiomed.2025.110723

GitHub citation: 


## Execution Order

### Step 1: Preprocessing
**Script:** `Shrinkage_preprocessing.py`
Run this first. It uses the ASD-GraphNet logic to map subjects and then computes OAS Shrinkage covariance matrices.
* **Output:** Generates `finalized_corr_shrinkage.pkl`.
* **Note:** This pickle file is the required input for all other analysis scripts.

### Step 2: Analysis & Ensembles
Once the `.pkl` file exists, you can run the following:
* **`Multiatlas.py`**: Runs the full ensemble with soft-voting and meta-classification (stacking) across CC200, AAL, and DOS160 atlases.
* **`shrinkage_tangent_rawandperm.py`**: Baseline Tangent Space models (SVM/LogReg) with site-stratified permutation testing.
* **`shrinkage_weighted_curvature.py`**: Extracts graph-theoretical features using Weighted Forman-Ricci Curvature (WFRC).
* **`shrinkage_alignment_scaling.py`**: Tests harmonization methods like Procrustes Alignment and NeuroHarmonize (ComBat) as well as StandardScaler and mean centering.
* **`riemann_dimensionalreduction.py`**: Evaluates PCA and ANOVA feature selection within the Tangent Space.
* **`FgMDM_shrinkage.py`**: Direct Riemannian classification.