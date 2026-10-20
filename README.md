# AUTOMATED LEUKEMIA DETECTION SYSTEM

## ABSTRACT & PROJECT ARCHITECTURE
An advanced, end-to-end deep learning web application engineered for the binary classification of microscopic blood smear cell images as ALL (Acute Lymphoblastic Leukemia - Malignant) versus HEM (Non-malignant / Normal) utilizing the Kaggle C-NMC dataset. Early and precise hematological screening is critical for effective clinical intervention. This repository houses an interactive decision-support system built around three distinct deep learning architectures, incorporating robust preprocessing, class-weight optimization, and automated single-run inference hosted via an interactive Streamlit user interface.

## EMPIRICAL MODEL PERFORMANCE & EVALUATION
The models were rigorously evaluated on an independent test dataset split, yielding the following performance metrics:

* Model 1: Baseline Custom CNN
  - Test Accuracy: 78.64%
  - Test Loss: 0.5567
  - ROC-AUC Score: 0.7810
  - ALL Recall (Sensitivity): 90.00%

* Model 2: Deeper Custom CNN
  - Test Accuracy: 77.85%
  - Test Loss: 0.5040
  - ROC-AUC Score: 0.7899
  - ALL Recall (Sensitivity): 87.00%

* Model 3: MobileNetV2 (Transfer Learning)
  - Test Accuracy: 79.23%
  - Test Loss: 0.4846
  - ROC-AUC Score: 0.7789
  - ALL Recall (Sensitivity): 95.00%

### Key Analytical Findings:
* Optimization Limitations in Deep Custom Models: Increasing network depth from Model 1 to Model 2 yielded marginal performance drops due to optimization ceilings and spatial information loss during aggressive max-pooling.
* Transfer Learning Efficacy: Model 3 (MobileNetV2 utilizing pre-trained ImageNet weights) demonstrated superior generalization, achieving an optimal 95% sensitivity (Recall) for malignant ALL cell detection, minimizing false negatives in high-stakes diagnostic screening.

## TECHNICAL STACK & DEPENDENCIES
* Programming Language: Python
* Deep Learning Framework: TensorFlow / Keras
* Interface & Deployment: Streamlit Community Cloud
* Data Processing & Utilities: NumPy, Pillow, gdown (for cloud-based artifact retrieval)

## REPOSITORY STRUCTURE
leukemia-detection-system/
- app.py: Streamlit web application & inference pipeline
- requirements.txt: Production dependency manifest
- .gitignore: Exclusion rules for binary weights and caches
- README.md: Technical documentation

## LOCAL INSTALLATION & EXECUTION
1. Clone the Repository:
   git clone https://github.com/Abh1goldy/leukemia-detection-system.git
   cd leukemia-detection-system

2. Install Environment Dependencies:
   pip install -r requirements.txt

3. Configure Model Checkpoints:
   Ensure your compiled .keras artifact files (model_1_basic.keras, model_2_good.keras, and model_3_best.keras) reside in the root directory. (Note: For cloud deployments, the application dynamically pulls assets via integrated cloud storage fetchers).

4. Initialize the Application:
   streamlit run app.py

## CLINICAL & ACADEMIC DISCLAIMER
This software artifact is engineered strictly for academic evaluation and engineering research demonstration. It serves as an auxiliary decision-support prototype and must not be utilized for autonomous clinical diagnosis or primary medical evaluation.
