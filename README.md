# Cassava Leaf Disease Classification Using Deep Learning and Classical Machine Learning

## Overview

This project investigates the performance of deep learning and classical machine learning approaches for automated cassava leaf disease classification.

The primary objective is to compare three Convolutional Neural Network (CNN) architectures — **ResNet50, EfficientNetB0 and MobileNetV2** — against traditional machine learning classifiers using handcrafted image features.

The project also investigates whether class weighting improves classification performance on an imbalanced dataset, particularly for minority disease categories.

In addition to predictive performance, the comparison considers:
- Overall classification accuracy and macro-F1 score.
- Per-class precision, recall and F1-score.
- Confusion matrices and misclassification patterns.
- Statistical significance of differences between models.
- Model size and inference latency.
- Feature extraction time for classical machine learning approaches.
- Model interpretability using Grad-CAM.

## Dataset

The dataset contains approximately **21,400 cassava leaf images** distributed across five classes:

| Class | Description |
|---|---|
| Cassava___bacterial_blight | Bacterial blight |
| Cassava___brown_streak_disease | Brown streak disease |
| Cassava___green_mottle | Green mottle |
| Cassava___mosaic_disease | Mosaic disease |
| Cassava___healthy | Healthy cassava leaves |

The dataset exhibits class imbalance, with mosaic disease representing the largest proportion of images and bacterial blight representing the smallest.

### Dataset splitting

A reproducible data-splitting strategy is used to ensure that all models are evaluated on the same images.

- **Training:** approximately 64%
- **Validation:** approximately 16%
- **Testing:** approximately 20%

The test split is created using a seeded shuffle performed separately for each class. The remaining training pool is subsequently divided into training and validation subsets.

The same image manifests are used for both CNN and classical machine learning experiments to maintain consistency and prevent data leakage.

## Methodology

### 1. Deep Learning Models

Three CNN architectures are investigated using transfer learning:

| Architecture | Description |
|---|---|
| ResNet50 | Residual convolutional neural network |
| EfficientNetB0 | Efficient compound-scaled convolutional neural network |
| MobileNetV2 | Lightweight convolutional neural network using inverted residual blocks |

All three architectures follow the same two-phase transfer-learning procedure.

**Phase 1 – Feature Extraction**

- Load the pretrained CNN backbone.
- Freeze the backbone layers.
- Train a newly added classification head.

**Phase 2 – Fine-Tuning**

- Unfreeze selected backbone layers.
- Continue training using a lower learning rate.
- Allow pretrained features to adapt to cassava leaf images.

Early stopping and model checkpointing are used to retain the best-performing weights.

#### Class-weighted EfficientNetB0

An additional EfficientNetB0 experiment incorporates class weighting to investigate the impact of class imbalance on model performance.

Class weights are calculated using the training data only:

\[
w_c = \frac{N}{K \times n_c}
\]

Where:

- \(N\) is the total number of training images.
- \(K\) is the number of classes.
- \(n_c\) is the number of images belonging to class \(c\).

This assigns greater importance to minority classes during training.

The original ResNet50, EfficientNetB0 and MobileNetV2 experiments remain unweighted, allowing the class-weighted EfficientNetB0 to be evaluated as a separate experiment.

### 2. Classical Machine Learning Models

Three classical classifiers are investigated:

- Support Vector Machine (SVM) with an RBF kernel.
- Random Forest.
- XGBoost.

Unlike CNNs, these algorithms require numerical feature representations rather than raw image inputs.

Therefore, handcrafted image features are extracted from the images identified by the shared manifests.

#### Handcrafted Feature Extraction

All images are resized to **256 × 256 pixels** before feature extraction.

The following features are extracted:

| Feature | Purpose |
|---|---|
| RGB Histograms | Capture colour distributions across RGB channels. |
| HSV Histograms | Represent hue, saturation and value distributions. |
| Local Binary Patterns (LBP) | Capture local texture patterns. |
| Gray-Level Co-occurrence Matrix (GLCM) | Extract texture properties such as contrast, homogeneity and correlation. |
| Histogram of Oriented Gradients (HOG) | Capture edge directions and structural patterns. |

The extracted features are concatenated into a single feature vector for each image.

Feature matrices are cached as NumPy `.npy` files to avoid repeatedly extracting features during subsequent experiments.

A `StandardScaler` is incorporated into each model pipeline to standardise the feature representations.

#### Hyperparameter Tuning

Hyperparameters are selected using validation macro-F1.

The selected configurations are subsequently evaluated using five-fold stratified cross-validation on the combined training and validation datasets.

The final classical models are refitted on training and validation data before evaluation on the held-out test set.

The test set is not used during hyperparameter selection.

## Evaluation Strategy

All models are evaluated using a common set of classification metrics.

### Classification Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correctly classified images. |
| Macro-F1 | Calculates the average F1-score across all classes, giving each class equal importance. |
| Balanced Accuracy | Measures average recall across the classes. |
| Precision | Measures the proportion of positive predictions that are correct for each class. |
| Recall | Measures the proportion of actual class instances correctly identified. |
| F1-score | Harmonic mean of precision and recall. |

Macro-F1 is particularly important because the dataset is imbalanced. A model that performs well on the dominant class may still perform poorly on minority disease categories.

### Statistical Analysis

Statistical comparisons are performed using the saved per-image predictions from the models.

The analysis includes:

- Bootstrap 95% confidence intervals for accuracy and macro-F1.
- Paired McNemar tests to compare classification disagreements between models.
- Holm correction to account for multiple statistical comparisons.

These analyses help determine whether differences in model performance are statistically meaningful.

### Model Interpretability

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to investigate the regions of cassava leaf images influencing CNN predictions.

Visualisations focus on:
- Correctly classified images.
- Bacterial blight images incorrectly predicted as healthy, where available.

EfficientNetB0 is the primary model for visualisation, with ResNet50 and MobileNetV2 included for comparison where checkpoints are available.

### Computational Efficiency

Model efficiency is investigated alongside predictive performance.

For the CNN models, the following measurements are collected:

- Total trainable and non-trainable parameters.
- Saved `.keras` checkpoint size.
- CPU inference latency using batch size 1.

For classical machine learning models, the analysis considers:

- Feature extraction latency.
- Model inference latency.
- Training and fitting time.

Feature extraction and model inference are measured separately to distinguish image preprocessing costs from classifier prediction costs.

## Project Structure

The notebook generates several artefacts during execution.

```text
cassava-leaf-disease-classification/
│
├── cassava_data/
│   ├── Cassava___bacterial_blight/
│   ├── Cassava___brown_streak_disease/
│   ├── Cassava___green_mottle/
│   ├── Cassava___mosaic_disease/
│   └── Cassava___healthy/
│
├── cassava_data_test/
│
├── artifacts/
│   ├── manifests/
│   │   ├── image_manifest.csv
│   │   ├── train_manifest.csv
│   │   ├── validation_manifest.csv
│   │   └── test_manifest.csv
│   │
│   ├── predictions/
│   │   └── *_test_predictions.csv
│   │
│   ├── classical/
│   │   ├── feature_cache/
│   │   └── results/
│   │
│   ├── gradcam/
│   ├── latency/
│   └── reproducibility/
│
├── results/
│   ├── bootstrap_test_intervals.csv
│   ├── mcnemar_paired_tests.csv
│   ├── master_model_results.csv
│   ├── master_accuracy_macro_f1.png
│   └── master_latency.png
│
├── checkpoints/
│   └── *.keras
│
├── model_final_analysis.ipynb
├── requirements.txt
└── README.md
```

*Note: The structure illustrates the main data and output directories. Actual checkpoint and result filenames depend on the notebook configuration.*

## Installation and Requirements

### Prerequisites

- Python 3.10 or later.
- Jupyter Notebook or JupyterLab.
- TensorFlow.
- OpenCV.
- Scikit-learn.
- Scikit-image.
- XGBoost.
- NumPy and Pandas.
- Matplotlib and Seaborn.

### Installation

Clone the repository:

```bash
git clone https://github.com/Al-amin1911/cassava-classification
cd cassava-leaf-disease-classification
```

## Running the Experiments

Open the Jupyter notebook:

```bash
jupyter notebook model_final_analysis.ipynb
```

Execute the notebook cells in order.

The general experimental workflow is:

1. Load and inspect the cassava leaf dataset.
2. Create the reproducible training, validation and test splits.
3. Perform image preprocessing and augmentation.
4. Train ResNet50, EfficientNetB0 and MobileNetV2.
5. Train the class-weighted EfficientNetB0 variant.
6. Export image manifests and per-image CNN predictions.
7. Extract handcrafted features for the classical models.
8. Train and tune SVM, Random Forest and XGBoost.
9. Perform five-fold cross-validation.
10. Evaluate all models on the held-out test set.
11. Calculate bootstrap confidence intervals and McNemar tests.
12. Generate Grad-CAM visualisations.
13. Measure model size, feature extraction time and CPU inference latency.
14. Generate the master comparison table and figures.
15. Save environment and reproducibility information.

Previously trained checkpoints and cached feature matrices can be reused where available, avoiding unnecessary retraining and feature extraction.

## Reproducibility

Reproducibility is supported through:

- Seeded dataset splitting.
- Consistent training, validation and test manifests.
- Fixed random states for cross-validation and statistical analysis.
- Saved model checkpoints.
- Cached handcrafted features.
- Saved per-image predictions.
- Recorded Python package versions using `pip freeze`.
- Saved experiment seeds and environment information.

The experiment outputs are stored under `artifacts/` and `results/` for subsequent analysis.

## Research Objective

The central research question is:

**How do transfer-learning-based CNN architectures compare with classical machine learning classifiers using handcrafted image features for cassava leaf disease classification, particularly when considering class imbalance, predictive performance and computational efficiency?**

The experiments are designed to investigate the trade-offs between:

1. Predictive performance across different disease categories.
2. The effectiveness of transfer learning compared with handcrafted feature extraction.
3. The impact of class weighting on minority-class classification.
4. Model complexity and computational efficiency.
5. The statistical significance and interpretability of model predictions.

## Acknowledgements

This project uses image-based machine learning techniques and established deep learning and classical machine learning architectures for agricultural disease classification.

Dataset source: `https://www.kaggle.com/datasets/nirmalsankalana/cassava-leaf-disease-classification?select=data`

---

**Project:** Cassava Leaf Disease Classification  
**Primary technologies:** Python, TensorFlow/Keras, Scikit-learn, OpenCV, Scikit-image, XGBoost  
**Experiment type:** Deep Learning and Classical Machine Learning Comparison
