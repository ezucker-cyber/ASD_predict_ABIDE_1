# ABIDE Classification: Geometric ML vs. Baselines

This repository contains the code for classifying Autism Spectrum Disorder (ASD) using resting-state fMRI data from the ABIDE-1 dataset. 

## Repository Structure

The codebase is split into two distinct analytical pipelines:

* **`preliminary_analysis/`**: Contains the initial codebase and standard baseline models used for early exploration.
* **`geometric_pipeline/`**: Contains the advanced models and frameworks detailed in the final report. This includes the Riemannian geometry Tangent Space mappings, Oracle Approximating Shrinkage (OAS), and the Multi-Atlas meta-classification ensembles.

## Setup & Data Requirements

**Crucially:** This pipeline does not include the raw fMRI data. You must download the preprocessed data from the repository of [Mina Zeraati and Amirehsan Davoodi](https://github.com/AmirDavoodi/ASD-GraphNet). You can follow the data access instructions provided in their repository.

### Attribution & Preprocessing

The logic for directory navigation, subject ID normalization, and linking the `.1D` atlas files to the phenotypic metadata is built upon and partially taken from the implementation in **ASD-GraphNet** by Mina Zeraati and Amirehsan Davoodi. 

For full details on the specific fMRI pre-processing steps used to generate these `.1D` files, please refer to their publication:

> Zeraati, M., & Davoodi, A. (2025). ASD-GraphNet: A novel graph learning approach for Autism Spectrum Disorder diagnosis using fMRI data. *Computers in Biology and Medicine*, 196(Pt B), 110723. https://doi.org/10.1016/j.compbiomed.2025.110723
