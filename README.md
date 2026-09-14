# Knee Classification Project

Machine-learning experiments for knee X-ray classification using handcrafted image features, deep-learning features, feature selection, and several classifiers.

## Project workflow

1. Prepare and label the knee X-ray images.
2. Extract image features such as HOG, GLCM, ORB, VGG, ResNet, SqueezeNet, and YOLO features.
3. Balance and split the feature data into training and test sets.
4. Select useful features with Harmony Search or fuzzy Harmony Search.
5. Train and compare machine-learning classifiers.

## Repository contents

- `feature_extraction.py`: Image preprocessing and feature extraction utilities.
- `data_processing.py`: Dataset splitting, balancing, and label-processing helpers.
- `harmony_feature_selection.py`: Harmony Search feature selection.
- `harmony_Fuzzy_feature_selection.py`: Fuzzy feature-selection implementation.
- `Main_ML.ipynb`: Main classification workflow.
- `Feature_Extraction_Stage.ipynb`: Feature-extraction workflow.
- `add_labels.py` and `check_labels.py`: Dataset label utilities.
- `README_Feature_Extraction.md`: Detailed feature-extraction reference.
- `Bin_Features/` and `final_Features/`: Generated feature datasets when available locally.

Generated datasets, model weights, NumPy files, and temporary folders are excluded by `.gitignore` because some are too large for normal GitHub storage.

## Requirements

Use Python 3.10 or newer when possible. Install the dependencies with:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The notebooks also require a Jupyter-compatible environment. In VS Code, select the `.venv` interpreter as the notebook kernel.

## Data format

The classification notebook expects feature CSV files with one row per image and a `label` column. The current notebook loads files from `Bin_Features/`, for example:

```text
Bin_Features/yolo_features_array.csv
Bin_Features/resnet_features_array.csv
```

Update `feature_type` and the input paths in the notebook when using a different feature set or dataset location.

The image dataset itself is not included in this repository. Keep local image data outside Git, or add its directory to `.gitignore` before committing it.

## Running the notebooks

1. Install the requirements.
2. Place or generate the feature CSV files in the expected local directories.
3. Open `Feature_Extraction_Stage.ipynb` if features need to be generated.
4. Open `Main_ML.ipynb` and run the cells in order.
5. Select the feature source by changing `feature_type` near the start of the notebook.

The notebooks can train multiple classifiers and display classification reports, confusion matrices, and accuracy comparisons. Training time depends on the number of samples, selected features, and enabled classifiers.

## Optional deep-learning models

VGG and YOLO extraction require additional model packages and model weights. TensorFlow is handled as an optional import by `feature_extraction.py`; YOLO weights such as `yolov11m.pt` are intentionally ignored by Git. Download model weights separately and keep them in the local workspace.

## Reproducibility

Several workflows use `random_state=42`. Results can still vary with package versions, hardware, model weights, and the exact generated feature files.

## Notes

- Do not commit patient-identifiable images or other private medical data.
- Do not commit large generated CSV files, model weights, or temporary notebook files.
- The spelling of existing directories such as `hormany_FS` is preserved to match the current project layout.
