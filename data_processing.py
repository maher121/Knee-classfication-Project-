import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter
import pandas as pd

def remove_rows_per_class(X, y, removal_ratio=0.2, random_state=42):
    """
    Remove a specified ratio of rows from each class
    
    Parameters:
    X: Feature array
    y: Label array
    removal_ratio: Ratio of rows to remove from each class (0.0 to 1.0)
    random_state: Random seed for reproducibility
    
    Returns:
    X_reduced, y_reduced: Arrays with rows removed from each class
    """
    np.random.seed(random_state)
    unique_classes = np.unique(y)
    indices_to_keep = []
    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        n_samples = len(class_indices)
        n_to_keep = int(n_samples * (1 - removal_ratio))
        selected_indices = np.random.choice(class_indices, size=n_to_keep, replace=False)
        indices_to_keep.extend(selected_indices)
    indices_to_keep = sorted(indices_to_keep)
    X_reduced = X[indices_to_keep]
    y_reduced = y[indices_to_keep]
    return X_reduced, y_reduced

def oversample_to_original_counts(X, y, original_class_counts, method='random', random_state=42):
    """
    Oversample the data back to original class counts
    
    Parameters:
    X: Feature array (reduced)
    y: Label array (reduced)
    original_class_counts: Dictionary with original count for each class
    method: 'random' for RandomOverSampler, 'smote' for SMOTE
    random_state: Random seed
    
    Returns:
    X_resampled, y_resampled: Oversampled arrays with original class counts
    """
    sampling_strategy = {}
    for class_label, original_count in original_class_counts.items():
        current_count = np.sum(y == class_label)
        if current_count < original_count:
            sampling_strategy[class_label] = original_count
    if not sampling_strategy:
        return X, y
    if method == 'random':
        sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    elif method == 'smote':
        sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    else:
        raise ValueError("Method must be 'random' or 'smote'")
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled

def custom_oversample_to_target(X, y, target_samples_per_class, random_state=42):
    """
    Oversample each class to reach a specific target number of samples
    
    Parameters:
    X: Feature array
    y: Label array
    target_samples_per_class: Target number of samples for each class
    random_state: Random seed
    
    Returns:
    X_resampled, y_resampled: Oversampled arrays
    """
    np.random.seed(random_state)
    unique_classes = np.unique(y)
    X_resampled_list = []
    y_resampled_list = []
    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        class_X = X[class_indices]
        class_y = y[class_indices]
        current_count = len(class_indices)
        if current_count < target_samples_per_class:
            n_to_generate = target_samples_per_class - current_count
            additional_indices = np.random.choice(len(class_X), size=n_to_generate, replace=True)
            additional_X = class_X[additional_indices]
            additional_y = class_y[additional_indices]
            final_X = np.vstack([class_X, additional_X])
            final_y = np.hstack([class_y, additional_y])
        else:
            final_X = class_X[:target_samples_per_class]
            final_y = class_y[:target_samples_per_class]
        X_resampled_list.append(final_X)
        y_resampled_list.append(final_y)
    X_resampled = np.vstack(X_resampled_list)
    y_resampled = np.hstack(y_resampled_list)
    return X_resampled, y_resampled

def process_data_for_accuracy(features_array, labels_array, 
                             removal_ratio=0.2, 
                             oversampling_method='random',
                             random_state=42):
    """
    Complete pipeline: remove rows from each class and then oversample back to original counts
    
    Parameters:
    features_array: Original feature array
    labels_array: Original label array
    removal_ratio: Ratio of samples to remove from each class
    oversampling_method: 'random' or 'smote'
    random_state: Random seed
    
    Returns:
    X_final, y_final: Processed arrays with original class distribution
    """
    original_class_counts = dict(Counter(labels_array))
    X_reduced, y_reduced = remove_rows_per_class(features_array, labels_array, 
                                               removal_ratio, random_state)
    X_final, y_final = oversample_to_original_counts(X_reduced, y_reduced, 
                                                   original_class_counts, 
                                                   oversampling_method, random_state)
    X_final, y_final = shuffle(X_final, y_final, random_state=random_state)
    return X_final, y_final
