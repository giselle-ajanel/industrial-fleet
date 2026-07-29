# Industrial Fleet Reliability & Fault Diagnostics System

An end-to-end machine learning pipeline built to predict component failures in heavy-duty commercial vehicles using tree-based ensemble methods. This project focuses on handling highly skewed data, severe missingness, and cost-sensitive classification on the Scania APS (Air Pressure System) Failure dataset.


## Overview

The primary goal of this system is to identify failures in the Air Pressure System (APS) of heavy Scania trucks to minimize maintenance costs and operational downtime. 

Predictive maintenance in fleet management requires handling cost asymmetry:
* **False Positives (Type I):** Costs associated with unnecessary truck inspection.
* **False Negatives (Type II):** Severe costs resulting from a breakdown on the road.

This project implements machine learning strategies optimized to handle severe class imbalance and feature sparsity.


## Dataset

The analysis utilizes the **APS Failure at Scania Trucks** dataset:
* **Training Set:** 60,000 observations across 171 features
* **Testing Set:** 16,000 observations across 171 features
* **Target Variable (`class`):** Binary (`pos` for APS component failure, `neg` for non-APS failures)
* **Missing Data:** High concentration of string missing values (`'na'`) across multiple feature bins.

## Data Preprocessing & EDA

1. **Missing Value Imputation:**
   - Identified string missing indicators (`'na'`) and converted them to `np.nan`.
   - Applied **Median Imputation** across numeric features to maintain robustness against extreme feature skewness and heavy outliers.

2. **Feature Dispersion Analysis:**
   - Evaluated feature variability using the Coefficient of Variation ($CV$):
     $$CV = \frac{\sigma}{\mu}$$

3. **Feature Interdependence:**
   - Generated a full $170 \times 170$ feature correlation matrix to analyze multi-collinearity and redundancy across sensor feature groups (`aa_000` through `eg_000`).


## Model Framework

The pipeline evaluates and hyperparameter-tunes ensemble architectures using stratified cross-validation:

* **Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)**
* **XGBoost Classifier (`xgboost.XGBClassifier`)**
* **Hyperparameter Optimization:** `GridSearchCV` combined with `StratifiedKFold` to preserve minority class ratios during fold splitting.
* **Evaluation Metrics:** Receiver Operating Characteristic (ROC-AUC), Confusion Matrix breakdown, Sensitivity/Recall prioritization, and Accuracy score.


## Installation & Setup

### Prerequisites
* Python 3.10+
* Conda or Virtualenv

### Dependencies
Install the required Python packages:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost
