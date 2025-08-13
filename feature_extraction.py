import cv2
import numpy as np
from skimage.feature import hog, graycomatrix, graycoprops
from skimage import data, exposure
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Handle optional dependencies
try:
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.vgg16 import preprocess_input
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. VGG16 features will not be available.")

try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: Ultralytics/YOLO not available. YOLO features will not be available.")

# ============================================================================
# STANDALONE FUNCTIONS FOR EASY IMPORT
# ============================================================================

def extract_yolov8_features(img, model_path='yolov8n.pt', model=None):
    """
    Extract features using the latest YOLOv8 model from Ultralytics.
    Args:
        img (numpy.ndarray): Input image (RGB)
        model_path (str): Path to YOLOv8 model weights (default: 'yolov8n.pt')
        model: Pre-loaded YOLO model (optional)
    Returns:
        numpy.ndarray: YOLOv8 features (flattened output tensor)
    """
    if not YOLO_AVAILABLE:
        print("Warning: Ultralytics YOLO not available. Returning zeros for YOLOv8 features.")
        return np.zeros(1000)
    if model is None:
        model = YOLO(model_path)
    # Resize image to model's expected input size (default 640x640 for YOLOv8)
    img_resized = cv2.resize(img, (640, 640))
    # Run inference
    results = model(img_resized, verbose=False)
    # Extract features from the output tensor
    # Use the first result's output (usually a torch tensor)
    try:
        output = results[0].boxes.data.detach().cpu().numpy().flatten()
        # Pad or truncate to fixed size
        if len(output) > 1000:
            output = output[:1000]
        elif len(output) < 1000:
            output = np.pad(output, (0, 1000 - len(output)), 'constant')
    except Exception:
        output = np.zeros(1000)
    return output

def extract_yolov8_features_from_image(img_path, model_path='yolov8n.pt'):
    """
    Convenience function to extract YOLOv8 features from an image path.
    Args:
        img_path (str): Path to the image file
        model_path (str): Path to YOLOv8 model weights
    Returns:
        numpy.ndarray: YOLOv8 features
    """
    img = preprocess_knee(img_path, image_size=(640, 640))
    return extract_yolov8_features(img, model_path=model_path)
def preprocess_knee(img_path, image_size=(224, 224)):
    """
    Preprocess knee X-ray image for feature extraction.
    
    Args:
        img_path (str): Path to the image file
        image_size (tuple): Target size for resizing (width, height)
    
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_REDUCED_GRAYSCALE_2)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image
    img = cv2.resize(img, image_size)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Apply Gaussian blur
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Apply sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    
    return img

def extract_hog_features(img, orientations=6, pixels_per_cell=(16, 16), cells_per_block=(1, 1)):
    """
    Extract HOG (Histogram of Oriented Gradients) features from image.
    
    Args:
        img (numpy.ndarray): Input image
        orientations (int): Number of orientation bins
        pixels_per_cell (tuple): Size of cells in pixels
        cells_per_block (tuple): Number of cells in each block
    
    Returns:
        tuple: (HOG features, HOG visualization image)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Extract HOG features
    features, hog_image = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                            cells_per_block=cells_per_block, visualize=True, block_norm='L2-Hys')
    
    return features, hog_image

def extract_glcm_features(img, distances=[1], angles=[0, 45, 90, 135]):
    """
    Extract GLCM (Gray Level Co-occurrence Matrix) features from image.
    
    Args:
        img (numpy.ndarray): Input image
        distances (list): List of pixel pair distance offsets
        angles (list): List of angles in degrees
    
    Returns:
        numpy.ndarray: GLCM features
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Calculate GLCM
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    # Extract GLCM properties
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    
    # Combine all features
    glcm_features = np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation])
    
    return glcm_features

def extract_orb_features(img, nfeatures=500):
    """
    Extract ORB (Oriented FAST and Rotated BRIEF) features from image.
    
    Args:
        img (numpy.ndarray): Input image
        nfeatures (int): Maximum number of features to retain
    
    Returns:
        numpy.ndarray: ORB features
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=nfeatures)
    
    # Detect keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    if descriptors is None:
        # If no keypoints found, return zeros
        return np.zeros(nfeatures)
    
    # Flatten descriptors and pad/truncate to fixed size
    descriptors_flat = descriptors.flatten()
    if len(descriptors_flat) > nfeatures:
        descriptors_flat = descriptors_flat[:nfeatures]
    elif len(descriptors_flat) < nfeatures:
        descriptors_flat = np.pad(descriptors_flat, (0, nfeatures - len(descriptors_flat)), 'constant')
    
    return descriptors_flat

def extract_vgg_features(img, model=None):
    """
    Extract VGG16 features from image.
    
    Args:
        img (numpy.ndarray): Input image
        model: Pre-loaded VGG16 model (optional)
    
    Returns:
        numpy.ndarray: VGG16 features
    """
    if not TENSORFLOW_AVAILABLE:
        print("Warning: TensorFlow not available. Returning zeros for VGG features.")
        return np.zeros(512)  # VGG16 features are typically 512-dimensional
    
    if model is None:
        model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    
    # Preprocess image for VGG16
    img_vgg = cv2.resize(img, (224, 224))
    img_vgg = image.img_to_array(img_vgg)
    img_vgg = np.expand_dims(img_vgg, axis=0)
    img_vgg = preprocess_input(img_vgg)
    
    # Extract features
    features = model.predict(img_vgg, verbose=0)
    
    return features.flatten()

def extract_yolo_features(img, model=None):
    """
    Extract YOLO features from image.
    
    Args:
        img (numpy.ndarray): Input image
        model: Pre-loaded YOLO model (optional)
    
    Returns:
        numpy.ndarray: YOLO features
    """
    if not YOLO_AVAILABLE:
        print("Warning: YOLO not available. Returning zeros for YOLO features.")
        return np.zeros(1000)  # YOLO features are typically 1000-dimensional
    
    if model is None:
        model = YOLO('yolov8n.pt')
    
    # Resize image to model's expected input size (default 640x640 for YOLOv8)
    img_resized = cv2.resize(img, (640, 640))
    # Run inference
    results = model(img_resized, verbose=False)
    # Extract features from the output tensor
    # Use the first result's output (usually a torch tensor)
    try:
        output = results[0].boxes.data.detach().cpu().numpy().flatten()
        # Pad or truncate to fixed size
        if len(output) > 1000:
            output = output[:1000]
        elif len(output) < 1000:
            output = np.pad(output, (0, 1000 - len(output)), 'constant')
    except Exception:
        output = np.zeros(1000)
    
    return output

def extract_all_features(img, vgg_model=None, yolo_model=None):
    """
    Extract all features from an image.
    
    Args:
        img (numpy.ndarray): Input image
        vgg_model: Pre-loaded VGG16 model (optional)
        yolo_model: Pre-loaded YOLO model (optional)
    
    Returns:
        tuple: (all_features, feature_dict)
    """
    # Extract different types of features
    hog_features, hog_image = extract_hog_features(img)
    glcm_features = extract_glcm_features(img)
    orb_features = extract_orb_features(img)
    vgg_features = extract_vgg_features(img, vgg_model)
    yolo_features = extract_yolo_features(img, yolo_model)
    
    # Combine all features
    all_features = np.concatenate([
        hog_features,
        glcm_features,
        orb_features,
        vgg_features,
        yolo_features
    ])
    
    return all_features, {
        'hog': hog_features,
        'glcm': glcm_features,
        'orb': orb_features,
        'vgg': vgg_features,
        'yolo': yolo_features,
        'hog_image': hog_image
    }

def extract_resnet50_features(img, model=None):
    """
    Extract ResNet50 features from an image.

    Args:
        img (numpy.ndarray): Input image (RGB, shape (H, W, 3))
        model: Pre-loaded ResNet50 model (optional)

    Returns:
        numpy.ndarray: ResNet50 features (2048-dimensional)
    """
    try:
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
        from tensorflow.keras.preprocessing import image as keras_image
    except ImportError:
        print("Warning: TensorFlow/Keras not available. Returning zeros for ResNet50 features.")
        return np.zeros(2048)

    if model is None:
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    # Resize and preprocess for ResNet50
    img_resnet = cv2.resize(img, (224, 224))
    img_resnet = keras_image.img_to_array(img_resnet)
    img_resnet = np.expand_dims(img_resnet, axis=0)
    img_resnet = resnet_preprocess(img_resnet)
    features = model.predict(img_resnet, verbose=0)
    return features.flatten()

