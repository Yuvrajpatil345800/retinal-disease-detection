# RetinaAI - AI-Based Diabetic Retinopathy Detection

RetinaAI is a Streamlit research prototype for classifying retinal fundus images into five diabetic retinopathy (DR) severity categories using a trained Vision Transformer (ViT). It is intended for academic demonstration and model-development workflows only.

> **Medical disclaimer:** This application is not clinically validated and is not a medical device. It must not be used to diagnose, treat, or make decisions about any patient. Always consult a qualified eye-care professional.

## Project overview

Diabetic retinopathy can cause progressive vision loss if it is not detected and managed promptly. RetinaAI demonstrates how a deep-learning image classifier can process retinal fundus photographs and return a predicted DR severity class with model probabilities.

## Problem statement

Reviewing retinal images requires trained clinical expertise. This project investigates an AI-assisted workflow that classifies a single fundus image into one of five DR severity categories. It does not replace clinical assessment.

## Objectives

- Load a trained five-class Vision Transformer checkpoint.
- Accept JPG, JPEG, and PNG retinal fundus images in a Streamlit interface.
- Apply the same preprocessing used by the inference model.
- Display the predicted class, confidence, and class probabilities.
- Store local prediction-history metadata and image previews in the enhanced UI.

## Key features

- Streamlit web interface for image upload and inference.
- Five DR severity classes.
- Vision Transformer inference through `load_model()` and `predict_image(model, image)`.
- 224 x 224 RGB preprocessing with ImageNet normalization.
- Basic app (`app.py`) and an enhanced multi-page UI (`app_v2.py`).
- Local prediction history in `data/history.json` and `data/history_images/` when using `app_v2.py`.
- Input validation for supported image formats and a 10 MB UI upload limit in `app_v2.py`.

## Technology stack

- Python
- Streamlit
- PyTorch and TorchVision
- timm
- Pillow
- Vision Transformer: `vit_base_patch16_224`

## Vision Transformer architecture

`model.py` creates a `timm` `vit_base_patch16_224` model with five output classes. Images are converted to RGB, resized to **224 x 224**, converted to tensors, and normalized with ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.

The checkpoint loader supports plain state dictionaries plus dictionaries containing `state_dict` or `model_state_dict`, and removes a `module.` prefix when present.

## Classification categories

1. No DR
2. Mild
3. Moderate
4. Severe
5. Proliferative DR

## Application workflow

1. Start the Streamlit application.
2. Upload one retinal fundus image.
3. The application converts the image to RGB and applies the inference transform.
4. The trained ViT produces logits and softmax probabilities.
5. The interface displays the predicted class, confidence, and probabilities for all five classes.
6. `app_v2.py` can save a local history record and preview image on the current machine.

## Project structure

```text
Retinal-DR-Detection/
├── app.py                         # Basic Streamlit interface
├── app_v2.py                      # Enhanced Streamlit interface
├── model.py                       # ViT loading and prediction functions
├── models/
│   └── best_vit_retinal_improved.pth  # Local checkpoint (not committed)
├── sample_images/                 # Optional local examples
├── data/                          # Local runtime history (not committed)
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation on Windows / VS Code

1. Clone the repository and open the folder in VS Code.
2. Open **Terminal** in VS Code.
3. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> PyTorch installation can differ by operating system, Python version, and CPU/GPU support. If the standard install fails, use the selector at [pytorch.org](https://pytorch.org/get-started/locally/) to obtain the compatible PyTorch command, then install the remaining packages from `requirements.txt`.

## Configure the trained model checkpoint

The checkpoint is intentionally excluded from ordinary Git commits because it is large. After cloning, obtain the approved checkpoint from the project owner or approved model storage, then place it at exactly:

```text
models/best_vit_retinal_improved.pth
```

Do not rename the file or alter `model.py`. If it is missing, the application will show a file-not-found error when inference starts.

## Run the application

Recommended enhanced interface:

```powershell
python -m streamlit run app_v2.py
```

Basic interface:

```powershell
python -m streamlit run app.py
```

Streamlit prints the local URL in the terminal, commonly `http://localhost:8501`.

## Use image upload and prediction

1. Open the Streamlit URL in a browser.
2. Navigate to **Upload & Predict** when using `app_v2.py`.
3. Upload one JPG, JPEG, or PNG retinal fundus image.
4. Select **Analyze Image**.
5. Review the predicted class and probabilities.

Predictions are model outputs only, not confirmed diagnoses.

## Model evaluation and metrics

This repository does not currently include a verified full-dataset evaluation report or accuracy metrics. Do not claim a clinical accuracy figure based solely on individual predictions. Before reporting metrics, evaluate the checkpoint against a held-out, appropriately licensed test set and record methodology, class distribution, accuracy, precision, recall, F1 score, confusion matrix, and any calibration analysis.

## Limitations and future improvements

- The model runs on CPU in the current `model.py` configuration.
- The checkpoint is not included in Git and must be supplied separately.
- History is stored locally, not shared or synchronized.
- The application has no clinical validation, patient-data workflow, or regulatory approval.
- Future work may include reproducible evaluation scripts, calibrated confidence reporting, approved secure model storage, stronger input-quality checks, and privacy-preserving deployment controls.

## Deployment prerequisites

Before deploying, ensure the host supports the required Python/PyTorch version and has enough memory and disk space for the ViT checkpoint. Supply the checkpoint through approved private deployment storage or the platform's secure artifact mechanism; do not commit it to ordinary Git. Configure Streamlit deployment secrets outside source control when needed, and test the complete inference path after deployment.

## License and data

Add a license and document the provenance, license, and consent conditions of any training or evaluation data before public release. Do not upload patient-identifiable images or restricted datasets to a public repository.
