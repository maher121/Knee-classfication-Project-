# Utility to apply selected features (binary vector) to a dataset
import numpy as np
def apply_selected_features(X, selected_features):
    """Apply the selected features (binary vector) to the dataset."""
    selected_feature_indices = np.where(selected_features == 1)[0]
    X_selected = X[:, selected_feature_indices]
    return X_selected
"""
Harmony Search Feature Selection Module

This module provides the harmony_search_feature_selection function for feature selection.
Copy this file to your project and import the function as needed.
"""

import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def initialize_harmony_memory(num_agents, num_features, min_features=1):
    harmony_memory = np.random.randint(0, 2, size=(num_agents, num_features))
    for i in range(num_agents):
        if np.sum(harmony_memory[i]) < min_features:
            while np.sum(harmony_memory[i]) < min_features:
                zero_indices = np.where(harmony_memory[i] == 0)[0]
                idx_to_flip = np.random.choice(zero_indices)
                harmony_memory[i][idx_to_flip] = 1
    return harmony_memory

def compute_fitness(harmony, X_train, X_val, y_train, y_val, min_features=1):
    selected_features = np.where(harmony == 1)[0]
    if len(selected_features) < min_features:
        return 0.0
    X_train_subset = X_train[:, selected_features]
    X_val_subset = X_val[:, selected_features]
    try:
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train_subset, y_train)
        predictions = classifier.predict(X_val_subset)
        accuracy = accuracy_score(y_val, predictions)
        feature_penalty = 0.001 * len(selected_features) / X_train.shape[1]
        return accuracy - feature_penalty
    except Exception as e:
        print(f"Error in compute_fitness: {e}")
        return 0.0

def sort_agents(harmony_memory, fitness):
    sorted_indices = np.argsort(fitness)[::-1]
    return harmony_memory[sorted_indices].copy(), fitness[sorted_indices].copy()

def apply_selected_features(X, selected_features):
    """Apply the selected features (binary vector) to the dataset."""
    selected_feature_indices = np.where(selected_features == 1)[0]
    X_selected = X[:, selected_feature_indices]
    return X_selected


def harmony_search_feature_selection(X, y, num_agents=20, max_iter=100, HMCR=0.9, PAR=0.3,
                                  min_features=1, early_stopping=10):
    num_features = X.shape[1]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    harmony_memory = initialize_harmony_memory(num_agents, num_features, min_features)
    fitness = np.zeros(num_agents)
    for i in range(num_agents):
        fitness[i] = compute_fitness(
            harmony_memory[i], X_train, X_val, y_train, y_val, min_features
        )
    harmony_memory, fitness = sort_agents(harmony_memory, fitness)
    best_harmony = harmony_memory[0].copy()
    best_fitness = fitness[0]
    no_improvement_count = 0
    for iteration in range(max_iter):
        new_harmony = np.zeros(num_features, dtype=int)
        for feature_index in range(num_features):
            if np.random.rand() < HMCR:
                random_agent = random.randint(0, num_agents - 1)
                new_harmony[feature_index] = harmony_memory[random_agent, feature_index]
                if np.random.rand() < PAR:
                    new_harmony[feature_index] = 1 - new_harmony[feature_index]
            else:
                new_harmony[feature_index] = np.random.randint(0, 2)
        if np.sum(new_harmony) < min_features:
            zero_indices = np.where(new_harmony == 0)[0]
            features_to_add = min_features - np.sum(new_harmony)
            indices_to_flip = np.random.choice(
                zero_indices, size=int(features_to_add), replace=False
            )
            new_harmony[indices_to_flip] = 1
        new_fitness = compute_fitness(
            new_harmony, X_train, X_val, y_train, y_val, min_features
        )
        if new_fitness > fitness[-1]:
            harmony_memory[-1] = new_harmony.copy()
            fitness[-1] = new_fitness
            harmony_memory, fitness = sort_agents(harmony_memory, fitness)
            if new_fitness > best_fitness:
                best_harmony = new_harmony.copy()
                best_fitness = new_fitness
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        else:
            no_improvement_count += 1
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{max_iter}")
            print(f"Best Fitness: {best_fitness:.4f}")
            print(f"Selected Features: {np.sum(best_harmony)}/{num_features}")
        if no_improvement_count >= early_stopping:
            print(f"\nStopping early - No improvement for {early_stopping} iterations")
            break
    selected_features = np.where(best_harmony == 1)[0]
    print("\nFeature selection completed.")
    print(f"Best Fitness: {best_fitness:.4f}")
    print(f"Number of selected features: {len(selected_features)}")
    print(f"Selected feature indices: {selected_features}")
    return best_harmony, best_fitness
