# Methodology and Script Reference

This document outlines the methodological framework underlying the geometric pipeline and provides a technical summary of the execution scripts. It highlights the core theoretical concepts designed to process the ABIDE dataset.

## Part 1: Methodological Framework

### 1. Stratified 5-Fold Cross-Validation
Our cross-validation approach splits the entire dataset into five distinct groups. Each group acts as the test set exactly once while the models are trained on the remaining four, ensuring that performance is evaluated on all samples while keeping model training and tuning strictly independent of the test set.
* **Site and Label Stratification:** Rather than splitting the data randomly, we utilize a fixed, stratified setup. By concatenating the scanning site and diagnosis, we ensure each of the five folds maintains consistent proportions of Autism/Control labels and collection sites. This robust stratification guarantees consistency in performance measurements across all experiments.

### 2. Block Permutation Testing
Standard permutation tests, which randomly shuffle diagnostic labels across an entire dataset, fail to reflect site-specific base rates in multi-site studies. 
* **Implementation:** This pipeline implements a site-stratified block permutation. The ASD and Control labels are scrambled exclusively within each site's boundaries. This generates a null distribution that strictly retains the site imbalances of the observed data, providing a mathematically sound baseline for significance testing.

### 3. Oracle Approximating Shrinkage (OAS)
When parcellating fMRI data, the number of brain regions often approaches or exceeds the number of time-points, creating highly unstable empirical covariance matrices.
* **Implementation:** OAS applies a mathematical penalty to pull extreme off-diagonal covariance values toward the mean variance. This regularization guarantees that the resulting matrices are Symmetric Positive Definite (SPD), a strict requirement for subsequent Riemannian operations.

### 4. Riemannian Geometry & Tangent Space Mapping
Covariance matrices reside on a curved Riemannian manifold. Applying standard Euclidean classifiers (like SVMs) directly to correlation values assumes a flat geometry, which distorts the true topological distances between functional networks.
* **Implementation:** The pipeline calculates the Fréchet mean of the training fold matrices. It then projects the SPD matrices onto a flat hyperplane tangent to this mean using a matrix logarithm operation. In this Tangent Space, the data becomes Euclidean, allowing algorithms like Logistic Regression to operate without geometric distortion.

---

## Part 2: Script Summaries

The following scripts orchestrate the pipeline. They must be executed from the directory containing the primary `.pkl` file.

### `Shrinkage_preprocessing.py`
* **Function:**  It processes the raw `.1D` timeseries, maps them to the phenotypic CSV to assign labels and applies the OAS estimator. 
* **Output:** Generates `finalized_corr_shrinkage.pkl`, which serves as the foundational dataset for all subsequent analyses.

### `shrinkage_tangent_rawandperm.py`
* **Function:** Establishes the core geometric baselines. It evaluates Linear SVM, RBF SVM, and Logistic Regression operating on Tangent Space vectors within a nested 5-fold cross-validation loop, and executes the block permutation test.

### `FgMDM_shrinkage.py`
* **Function:** Bypasses the Tangent Space projection to perform direct classification on the manifold. It uses the Minimum Distance to Mean (MDM) algorithm to classify subjects based on their Riemannian geodesic distance to the class-conditional Fréchet means.

### `riemann_dimensionalreduction.py`
* **Function:** Evaluates feature compression strategies. It applies Principal Component Analysis (PCA) and ANOVA (SelectKBest) to the Tangent Space vectors.

### `shrinkage_alignment_scaling.py`
* **Function:** Evaluates multiple scaling and alignment techniques: Standard scaling, Mean Centering, Procrustes Alignment, and NeuroHarmonize (ComBat).

### `shrinkage_weighted_curvature.py`
* **Function:** Extracts graph-theoretic network topology. It applies thresholds to the connectivity matrices and separates positive and negative networks, subsequently extracting Weighted Forman-Ricci Curvature (WFRC) node features to model information flow bottlenecks.

### `Multiatlas.py`
* **Function:** The ensemble architecture. It trains three separate, independent base models—one for each of the AAL, DOS160, and CC200 atlases. It then combines their predictions to capture information across different spatial resolutions. This is accomplished via a Soft Voting ensemble (averaging the prediction probabilities of the three independent models) and a Meta-classifier ensemble (stacking the independent predictions as inputs into a secondary Logistic Regression model).