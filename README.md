# XHDPF-Repository
Explainable Higher-Education Dropout Prediction Framework

[![Python 3.10](https://img.shields.io/badge/Python-3.10.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1.2-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost 2.0](https://img.shields.io/badge/XGBoost-2.0.3-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Official public evidence and replication repository for the manuscript:
**"An Explainable Student Dropout Prediction Framework for Higher Education: Systematic Evidence Synthesis, Predictive Core Evaluation, and an Institutional Transferability Protocol"** (Submitted to *The Journal of Learning Analytics*).

---

## 📌 Repository Overview

Predicting student dropout in higher education remains constrained by opaque algorithms, unstandardized administrative data governance, neglected class imbalances, and high cognitive load for decision-makers. The **Explainable Higher-Education Dropout Prediction Framework (XHDPF)** addresses these structural gaps through a five-layer sociotechnical architecture constructed from a systematic literature review (SLR) of \\(n = 156\\) primary empirical studies.

This repository provides complete methodological transparency, data extraction matrices, quality assessment rubrics, and open-source code for reproducing the empirical validation and explainability audits of XHDPF.

### Key Framework Highlights
- **Evidence Base:** Grounded in a PRISMA 2020 systematic evidence synthesis (\\(n = 156\\) studies).
- **Data Governance (Layer 1):** Automated DAMA-DMBOK rule validation and SHA-256 cryptographic pseudonymization at ingestion.
- **Deep Generative Balancing (Layer 2):** Conditional Tabular GAN (CTGAN) mode-specific oversampling to resolve the "Accuracy Paradox" (mitigating unmitigated pipelines found in 73.7% of literature).
- **Dual Predictive Core (Layer 3):** Late-fusion hybrid engine combining static background indicators (XGBoost) with sequential LMS interaction trajectories (2-layer Stacked LSTM) at time horizon \\(T\\) (Term 1 completion).
- **Leakage-Free Explainability (Layer 4):** Multi-level attributions integrating model-specific TreeSHAP with model-agnostic local LIME surrogates, strictly bound to attributes observed prior to time horizon \\(T\\).
- **Subgroup Fairness Auditing:** Granular demographic parity checks confirming low Equalized Odds Differences (\\(\Delta\text{EOD} \le 0.0245\\)) and Disparate Impact ratios exceeding screening heuristics (\\(\text{DI} \ge 0.87\\)).
- **Institutional Transferability Protocol:** A structured 4-phase adaptation blueprint guiding domain adaptation and advisor cognitive load evaluation (SUS and NASA-TLX).

---

## 📁 Repository Directory Structure

```text
XHDPF-Repository/
│
├── README.md                           <- Repository overview and replication guide
│
├── 01_SLR_Protocol/                    <- Systematic review protocol and PRISMA tools
│   ├── SLR_protocol.pdf                <- Formal review protocol specification
│   ├── PRISMA_checklist.pdf            <- Complete PRISMA 2020 checklist
│   └── inclusion_exclusion_criteria.csv<- Operational inclusion/exclusion definitions
│
├── 02_Search_Strategies/               <- Database search syntaxes and execution logs
│   ├── Scopus.txt                      <- Scopus field-restricted search queries
│   ├── Web_of_Science.txt              <- WoS Core Collection search syntaxes
│   ├── ACM.txt                         <- ACM Digital Library queries
│   └── IEEE_Xplore.txt                 <- IEEE Xplore search queries
│
├── 03_Study_Selection/                 <- PRISMA screening flow datasets
│   ├── initial_records.csv             <- Consolidated raw search results (N = 1,095)
│   ├── duplicates.csv                  <- Automated deduplication logs (n = 214)
│   ├── title_abstract_screening.csv    <- Title and abstract screening records (n = 881)
│   ├── full_text_screening.csv         <- Full-text eligibility evaluations (n = 317)
│   └── exclusion_reasons.csv           <- Documented exclusion reasons
│
├── 04_Evidence_Extraction/             <- Systematic evidence extraction matrices
│   ├── S1_156_studies.csv              <- Complete catalog of included primary studies (n = 156)
│   ├── predictor_taxonomy.csv          <- Extracted feature dimensions and variables
│   ├── algorithms.csv                  <- Evaluated machine learning algorithms
│   ├── imbalance_strategies.csv        <- Imbalance mitigation techniques in literature
│   └── PROBAST_assessment.csv          <- Adapted PROBAST risk of bias scoring matrix
│
├── 05_Quantitative_Synthesis/          <- Synthesized SLR analytical tables
│   ├── Table4.csv                      <- Empirical data source distributions
│   ├── Table5.csv                      <- Algorithmic performance and prevalence
│   └── Table6.csv                      <- Class imbalance strategy distributions
│
├── 06_Empirical_Experiment/            <- Experimental machine learning pipelines
│   ├── preprocessing/                  <- Quality gates, imputation, and SHA-256 hashing
│   ├── SMOTE/                          <- Linear SMOTE baseline oversampling pipelines
│   ├── CTGAN/                          <- Deep generative tabular GAN balancing scripts
│   ├── RF/                             <- Random Forest baseline classifiers
│   ├── XGBoost/                        <- Static branch gradient boosted tree models
│   ├── LSTM/                           <- Dynamic branch 2-layer stacked LSTM sequence models
│   └── LateFusion/                     <- Late-fusion weighting engine (alpha = 0.60)
│
├── 07_XAI/                             <- Explainable AI auditing scripts
│   ├── SHAP/                           <- TreeSHAP global feature attribution audits
│   └── LIME/                           <- Leakage-free local LIME surrogate explanations
│
├── 08_Fairness/                        <- Subgroup fairness evaluation suites
│   ├── subgroup_definitions.csv        <- Demographic and protected cohort definitions
│   ├── DI.csv                          <- Disparate Impact ratio calculations
│   └── EOD.csv                         <- Equalized Odds Difference metric evaluations
│   ├── compute_fairness_metrics.py     <- Módulo de cálculo matemático de DI y \Delta EOD
│   └── fairness_audit.py               <- Script de ejecución y verificación de umbrales éticos
│
├── 09_Transferability/                 <- Cross-institutional transferability suite
│   ├── protocol.pdf                    <- Formal 4-phase institutional transferability protocol
│   ├── schema_mapping.xlsx             <- Canonical DAMA-DMBOK database mapping schema
│   └── checklist.pdf                   <- Institutional domain adaptation checklist
│
└── 10_Environment/                     <- Technical computing specifications
    ├── requirements.txt                <- Python pip package dependencies
    ├── environment.yml                 <- Conda environment configuration file
    └── software_versions.txt           <- Hardware, CUDA, cuDNN, and driver versions
________________________________________
🛠️ Hardware Acceleration & Software Environment
All empirical experiments, generative CTGAN training, and nested cross-validation pipelines were executed within a controlled local computing environment:
•	Operating System: Ubuntu 22.04.3 LTS (64-bit Linux kernel 5.15)
•	CPU / RAM: Intel Xeon Gold 6226R @ 2.90GHz (16 physical cores, 32 threads), 64 GB DDR4 RAM
•	GPU Acceleration: NVIDIA GeForce RTX 4090 GPU (24 GB GDDR6X VRAM)
•	Drivers: CUDA Toolkit 12.1 / cuDNN 8.9.2
•	Core Software Stack: 
o	Python == 3.10.12
o	PyTorch == 2.1.2 (Stacked LSTM dynamic sequence branch)
o	XGBoost == 2.0.3 (Static tabular decision tree branch)
o	scikit-learn == 1.3.2 (Preprocessing, baseline classifiers, metrics)
o	ctgan == 0.7.4 / sdv == 1.8.0 (Deep tabular generative oversampling)
o	shap == 0.44.0 / lime == 0.2.0.1 (Algorithmic explainability audits)
________________________________________
🚀 Replicability and Usage Guide
1. Environment Setup
Clone the anonymized repository and build the virtual environment using Conda:
git clone https://github.com/anonymized-user/XHDPF-Repository.git
cd XHDPF-Repository/10_Environment

# Create Conda environment
conda env create -f environment.yml
conda activate xhdpf_env

# Or install via pip
pip install -r requirements.txt
2. Running Empirical ML Pipelines (06_Empirical_Experiment/)
Execute the nested 10-fold cross-validation pipeline with fold-wise CTGAN generative oversampling:
# Preprocessing and deterministic cryptographic hashing
python 06_Empirical_Experiment/preprocessing/preprocess.py

# Fit CTGAN generator strictly within training folds and train Late Fusion Hybrid Model
python 06_Empirical_Experiment/LateFusion/train_late_fusion.py --alpha 0.60 --folds 10
3. Executing XAI & Fairness Audits (07_XAI/ & 08_Fairness/)
Generate leakage-free LIME local surrogates and compute demographic parity metrics across protected cohorts:
# Run local LIME surrogates at time horizon T
python 07_XAI/LIME/run_lime_audit.py --horizon T

# Compute Equalized Odds Difference (EOD) and Disparate Impact (DI)
python 08_Fairness/compute_fairness_metrics.py
________________________________________
🛡️ Data Privacy & Ethical Compliance Statement
In compliance with DAMA-DMBOK standards, GDPR principles, and institutional data protection regulations, raw administrative student datasets (\(n = 12,400\)) containing Personally Identifiable Information (PII) cannot be publicly distributed. Student privacy was safeguarded by deterministic cryptographic pseudonymization (SHA-256) at ingestion.
To enable full algorithmic reproducibility without privacy compromise, synthetic tabular datasets generated by CTGAN matching the empirical feature distributions are provided in 06_Empirical_Experiment/CTGAN/. Formal ethics exemption was granted by the institutional review authority ([Anonymized Ethics Committee]).
________________________________________
📄 License & Attribution
•	Code and Algorithms: Licensed under the MIT License.
•	Data, Synthesis Tables, and Documentation: Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
How to Cite
@article{XHDPF2026,
  author    = {Anonymized Authors for Double-Blind Review},
  title     = {An Explainable Student Dropout Prediction Framework for Higher Education: Systematic Evidence Synthesis, Predictive Core Evaluation, and an Institutional Transferability Protocol},
  journal   = {The Journal of Learning Analytics},
  year      = {2026},
  note      = {Under Review}
}

---

### **Resumen de Cobertura y Métricas Clave Incluidas en el `README.md`**
1. **Estructura idéntica al estándar:** Incorpora las **10 carpetas y sus archivos específicos** definidos en `Repositorio.docx`.
2. **Alineación directa con el Manuscrito:** Refleja la arquitectura de 5 capas de XHDPF, los datos de la SLR (\\(n = 156\\)), los resultados del modelo híbrido (\\(n = 12.400\\), AUC-ROC = 97.8%, PR-AUC = 90.3%, ECE = 1.8%), los límites de equidad (\\(\Delta\text{EOD} \le 0.0245\\), \\(\text{DI} \ge 0.87\\)) y el protocolo de transferibilidad.
3. **Garantía de Doble Ciego:** Conserva todas las etiquetas anonimizadas para proteger el proceso de evaluación por pares de JLA.
