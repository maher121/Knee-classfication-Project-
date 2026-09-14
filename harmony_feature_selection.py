import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed # 1. استيراد المكتبة للمعالجة المتوازية

def initialize_harmony_memory(num_agents, num_features, min_features=1):
    """Initialize the harmony memory with random solutions ensuring min features."""
    harmony_memory = np.random.randint(0, 2, size=(num_agents, num_features))
    for i in range(num_agents):
        # Ensure minimum features constraint
        if np.sum(harmony_memory[i]) < min_features:
            zero_indices = np.where(harmony_memory[i] == 0)[0]
            features_to_add = min_features - np.sum(harmony_memory[i])
            indices_to_flip = np.random.choice(
                zero_indices, size=int(features_to_add), replace=False
            )
            harmony_memory[i][indices_to_flip] = 1
    return harmony_memory

def compute_fitness(harmony, X_train, X_val, y_train, y_val, min_features, classifier):
    """Compute fitness for a single harmony using a pre-configured classifier."""
    selected_features = np.where(harmony == 1)[0]
    
    # Penalize if features are below minimum
    if len(selected_features) < min_features:
        return 0.0
    
    X_train_subset = X_train[:, selected_features]
    X_val_subset = X_val[:, selected_features]
    
    try:
        # We clone or fit the passed classifier instance
        # Note: To ensure thread safety in parallel execution, it's often better to create a new instance
        # or ensure the classifier resets. Here we create a lightweight one for speed.
        classifier.fit(X_train_subset, y_train)
        predictions = classifier.predict(X_val_subset)
        accuracy = accuracy_score(y_val, predictions)
        
        # Feature penalty term
        feature_penalty = 0.001 * len(selected_features) / X_train.shape[1]
        return accuracy - feature_penalty
    except Exception as e:
        print(f"Error in compute_fitness: {e}")
        return 0.0

def sort_agents(harmony_memory, fitness):
    """Sort agents based on fitness in descending order."""
    sorted_indices = np.argsort(fitness)[::-1]
    return harmony_memory[sorted_indices].copy(), fitness[sorted_indices].copy()

def apply_selected_features(X, selected_features):
    """Apply the selected features (binary vector) to the dataset."""
    selected_feature_indices = np.where(selected_features == 1)[0]
    X_selected = X[:, selected_feature_indices]
    return X_selected

def harmony_search_feature_selection(X, y, num_agents=20, max_iter=100, HMCR=0.9, PAR=0.3,
                                  min_features=1, early_stopping=10,
                                  n_estimators=50, max_depth=None, n_jobs=2):
    """
    Harmony Search Feature Selection with optimizations.
    
    Parameters:
    -----------
    n_estimators : int
        Number of trees in the random forest (lower is faster).
    max_depth : int
        Max depth of trees (limits overfitting and speeds up training).
    n_jobs : int
        Number of parallel jobs to run for initial population evaluation (-1 for all cores).
    """
    
    num_features = X.shape[1]
    
    # Split data once
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Initialize Harmony Memory
    harmony_memory = initialize_harmony_memory(num_agents, num_features, min_features)
    
    # Optimized Classifier for the SEARCH phase
    # n_jobs=1 is important here because we will parallelize the outer loop (initial population)
    search_classifier = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=42,
        n_jobs=1 
    )
    
    print(f"Evaluating initial population ({num_agents} agents) using {n_jobs} cores...")
    
    # 2. استخدام Parallel لتقييم المجتمع الأولي بالتوازي (تسريع كبير في البداية)
    fitness_list = Parallel(n_jobs=n_jobs)(
        delayed(compute_fitness)(
            agent, X_train, X_val, y_train, y_val, min_features, search_classifier
        ) for agent in harmony_memory
    )
    fitness = np.array(fitness_list)
    
    harmony_memory, fitness = sort_agents(harmony_memory, fitness)
    
    best_harmony = harmony_memory[0].copy()
    best_fitness = fitness[0]
    no_improvement_count = 0
    
    print("Starting Optimization Loop...")
    
    for iteration in range(max_iter):
        new_harmony = np.zeros(num_features, dtype=int)
        
        # Improvisation step
        for feature_index in range(num_features):
            if np.random.rand() < HMCR:
                random_agent = random.randint(0, num_agents - 1)
                new_harmony[feature_index] = harmony_memory[random_agent, feature_index]
                if np.random.rand() < PAR:
                    new_harmony[feature_index] = 1 - new_harmony[feature_index]
            else:
                new_harmony[feature_index] = np.random.randint(0, 2)
        
        # Ensure minimum features
        if np.sum(new_harmony) < min_features:
            zero_indices = np.where(new_harmony == 0)[0]
            features_to_add = min_features - np.sum(new_harmony)
            if len(zero_indices) >= features_to_add:
                indices_to_flip = np.random.choice(
                    zero_indices, size=int(features_to_add), replace=False
                )
                new_harmony[indices_to_flip] = 1
        
        # Evaluate new harmony (Sequential inside the main loop, but faster classifier)
        # Note: We pass a clone or a new instance implicitly if compute_fitness handles it,
        # but here we rely on the search_classifier being re-fit.
        # Ideally, create a new instance for thread safety if the loop was parallel.
        new_fitness = compute_fitness(
            new_harmony, X_train, X_val, y_train, y_val, min_features, search_classifier
        )
        
        # Update Harmony Memory
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
            print(f"Iteration {iteration + 1}/{max_iter} | Best Fitness: {best_fitness:.4f} | Features: {np.sum(best_harmony)}")
        
        if no_improvement_count >= early_stopping:
            print(f"\nStopping early - No improvement for {early_stopping} iterations")
            break
            
    print("\nFeature selection completed.")
    print(f"Best Fitness: {best_fitness:.4f}")
    print(f"Number of selected features: {np.sum(best_harmony)}")
    print(f"Selected feature indices: {np.where(best_harmony == 1)[0]}")
    
    return best_harmony, best_fitness