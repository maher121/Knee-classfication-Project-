# Feature Extraction Module

A comprehensive feature extraction module for knee X-ray image analysis, designed to work with the Fuzzy Harmony Search feature selection algorithm.

## Overview

This module provides both standalone functions and a class-based approach for extracting various types of features from knee X-ray images:

- **HOG (Histogram of Oriented Gradients)** features
- **GLCM (Gray Level Co-occurrence Matrix)** features  
- **ORB (Oriented FAST and Rotated BRIEF)** features
- **VGG16** deep learning features (optional)
- **YOLO** features (optional)

## Installation

### Required Dependencies

```bash
pip install opencv-python
pip install scikit-image
pip install scikit-learn
pip install numpy
pip install matplotlib
pip install tqdm
pip install seaborn
pip install xgboost
```

### Optional Dependencies

For deep learning features (VGG16 and YOLO):

```bash
pip install tensorflow
pip install ultralytics
pip install torch
```

## Quick Start

### Basic Usage with Standalone Functions

```python
from feature_extraction import (
    preprocess_knee,
    extract_hog_features,
    extract_glcm_features,
    extract_orb_features
)

# Preprocess an image
img = preprocess_knee('path/to/knee/image.jpg')

# Extract individual features
hog_features, hog_image = extract_hog_features(img)
glcm_features = extract_glcm_features(img)
orb_features = extract_orb_features(img)

# Extract all features at once
from feature_extraction import extract_all_features
all_features, feature_dict = extract_all_features(img)
```

### Class-Based Approach

```python
from feature_extraction import ImageFeatureExtractor

# Create feature extractor
extractor = ImageFeatureExtractor(image_size=(224, 224))

# Load pre-trained models (optional)
extractor.load_models()

# Extract features
img = extractor.preprocess_image('path/to/image.jpg')
features, feature_dict = extractor.extract_all_features(img)
```

## Available Functions

### Core Feature Extraction Functions

#### `preprocess_knee(img_path, image_size=(224, 224))`
Preprocesses knee X-ray images with:
- Resizing to specified dimensions
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gaussian blur
- Sharpening filter

**Parameters:**
- `img_path` (str): Path to the image file
- `image_size` (tuple): Target size for resizing (width, height)

**Returns:**
- `numpy.ndarray`: Preprocessed image

#### `extract_hog_features(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))`
Extracts HOG features from the image.

**Parameters:**
- `img` (numpy.ndarray): Input image
- `orientations` (int): Number of orientation bins
- `pixels_per_cell` (tuple): Size of cells in pixels
- `cells_per_block` (tuple): Number of cells in each block

**Returns:**
- `tuple`: (HOG features, HOG visualization image)

#### `extract_glcm_features(img, distances=[1], angles=[0, 45, 90, 135])`
Extracts GLCM features from the image.

**Parameters:**
- `img` (numpy.ndarray): Input image
- `distances` (list): List of pixel pair distance offsets
- `angles` (list): List of angles in degrees

**Returns:**
- `numpy.ndarray`: GLCM features

#### `extract_orb_features(img, nfeatures=500)`
Extracts ORB features from the image.

**Parameters:**
- `img` (numpy.ndarray): Input image
- `nfeatures` (int): Maximum number of features to retain

**Returns:**
- `numpy.ndarray`: ORB features

#### `extract_all_features(img, vgg_model=None, yolo_model=None)`
Extracts all types of features from the image.

**Parameters:**
- `img` (numpy.ndarray): Input image
- `vgg_model`: Pre-loaded VGG16 model (optional)
- `yolo_model`: Pre-loaded YOLO model (optional)

**Returns:**
- `tuple`: (all_features, feature_dict)

### Utility Functions

#### `apply_selected_features(X, selected_features)`
Applies feature selection to the dataset.

#### `evaluate_classifier(classifier, X_train, X_test, y_train, y_test, classifier_name="")`
Evaluates a classifier and prints results.

#### `prepare_data(features, labels)`
Prepares data for machine learning (encoding labels, splitting data).

#### `evaluate_all_classifiers(X_train, X_test, y_train, y_test)`
Evaluates multiple classifiers and returns results.

#### `plot_results_comparison(results)`
Plots comparison of classifier results.

#### `visualize_features(img, feature_dict)`
Visualizes extracted features.

### Convenience Functions

#### `extract_features_from_image(img_path, image_size=(224, 224))`
Convenience function to extract all features from a single image.

#### `extract_features_from_dataset(dataset_path, image_size=(224, 224))`
Convenience function to extract features from entire dataset.

#### `run_complete_analysis(dataset_path, image_size=(224, 224))`
Runs complete analysis pipeline: feature extraction, classification, and evaluation.

## Integration with Fuzzy Harmony Search

The module is designed to work seamlessly with the Fuzzy Harmony Search feature selection algorithm:

```python
from feature_extraction import (
    extract_features_from_dataset,
    prepare_data,
    apply_selected_features,
    evaluate_all_classifiers
)
from fuzzy_harmony_search import FuzzyHarmonySearchFS

# Extract features from dataset
features, labels, image_paths = extract_features_from_dataset('path/to/dataset')

# Prepare data
X_train, X_test, y_train, y_test, label_encoder = prepare_data(features, labels)

# Initialize Fuzzy Harmony Search
fs = FuzzyHarmonySearchFS(
    num_agents=30,
    max_iter=100,
    HMCR=0.85,
    PAR=0.35,
    weights=(0.7, 0.2, 0.1),
    random_state=42,
    verbose=True,
    membership_type='gg',
    n_clusters=3
)

# Perform feature selection
fs.fit(X_train, y_train)

# Apply selected features
X_train_selected = apply_selected_features(X_train, fs.best_harmony)
X_test_selected = apply_selected_features(X_test, fs.best_harmony)

# Evaluate classifiers
results = evaluate_all_classifiers(X_train_selected, X_test_selected, y_train, y_test)
```

## Example Workflows

### 1. Single Image Analysis

```python
from feature_extraction import extract_features_from_image, visualize_features

# Extract features from single image
features, feature_dict, img = extract_features_from_image('path/to/image.jpg')

# Visualize features
visualize_features(img, feature_dict)

print(f"Total features extracted: {len(features)}")
```

### 2. Dataset Analysis

```python
from feature_extraction import run_complete_analysis

# Run complete analysis on dataset
results = run_complete_analysis('path/to/dataset')

# Access results
print(f"Features shape: {results['features'].shape}")
print(f"Number of images: {len(results['image_paths'])}")
print(f"Best classifier: {max(results['results'], key=lambda k: results['results'][k]['test_accuracy'])}")
```

### 3. Custom Feature Extraction

```python
from feature_extraction import ImageFeatureExtractor

# Create custom feature extractor
extractor = ImageFeatureExtractor(image_size=(512, 512))

# Load dataset
features, labels, image_paths = extractor.load_dataset('path/to/dataset')

# Normalize features
features_normalized = extractor.normalize_features(features)

# Apply PCA for dimensionality reduction
features_pca = extractor.apply_pca(features_normalized)
```

## Feature Dimensions

The module extracts the following feature dimensions:

- **HOG features**: 26,244 dimensions (default parameters)
- **GLCM features**: 20 dimensions (4 angles × 5 properties)
- **ORB features**: 500 dimensions (configurable)
- **VGG16 features**: 512 dimensions (when TensorFlow is available)
- **YOLO features**: 1,000 dimensions (when YOLO is available)

**Total**: ~28,276 dimensions (when all features are available)

## Error Handling

The module handles missing dependencies gracefully:

- If TensorFlow is not available, VGG16 features return zeros
- If YOLO is not available, YOLO features return zeros
- All other features (HOG, GLCM, ORB) work without additional dependencies

## Performance Tips

1. **Batch Processing**: Use `extract_features_from_dataset()` for processing multiple images
2. **Model Caching**: Load VGG16 and YOLO models once and reuse them
3. **Memory Management**: For large datasets, consider processing in batches
4. **Parallel Processing**: The module can be easily extended for parallel processing

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all required dependencies are installed
2. **Memory Issues**: Reduce image size or process in smaller batches
3. **Slow Performance**: Consider using only essential features (HOG + GLCM)
4. **Model Loading**: Ensure model files are in the correct location

### Getting Help

If you encounter issues:

1. Check that all dependencies are installed correctly
2. Verify image paths are correct
3. Ensure sufficient memory for processing
4. Check the console output for warning messages

## Contributing

To extend the module:

1. Add new feature extraction functions
2. Update the `extract_all_features()` function
3. Add corresponding visualization code
4. Update documentation

## License

This module is part of the Knee Project and follows the same license terms. 