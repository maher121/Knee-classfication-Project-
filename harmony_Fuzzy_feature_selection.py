import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy
from scipy.spatial.distance import mahalanobis
from sklearn.mixture import GaussianMixture
import time
from tqdm import tqdm

class GKMembreship:
    """Gustafson-Kessel inspired membership functions"""

    def __init__(self, n_clusters=3, m=2, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.m = m  # Fuzziness coefficient
        self.max_iter = max_iter
        self.random_state = random_state
        self.centers_ = None
        self.covariances_ = None
        self.memberships_ = None

    def fit(self, X):
        """Learn cluster centers and covariance matrices"""
        n_samples, n_features = X.shape

        # Initialize centers randomly
        rng = np.random.RandomState(self.random_state)
        self.centers_ = X[rng.choice(n_samples, self.n_clusters, replace=False)]

        # Initialize covariance matrices as identity
        self.covariances_ = [np.eye(n_features) for _ in range(self.n_clusters)]

        for _ in range(self.max_iter):
            # Step 1: Compute distances using Mahalanobis distance
            distances = np.zeros((n_samples, self.n_clusters))
            for i in range(self.n_clusters):
                try:
                    inv_cov = np.linalg.inv(self.covariances_[i])
                except np.linalg.LinAlgError:
                    inv_cov = np.linalg.pinv(self.covariances_[i])

                for j in range(n_samples):
                    diff = X[j] - self.centers_[i]
                    distances[j,i] = np.sqrt(diff.T @ inv_cov @ diff)

            # Step 2: Update memberships
            power = 2/(self.m-1)
            with np.errstate(divide='ignore', invalid='ignore'):
                memberships = 1 / (distances ** power)
                memberships = memberships / np.sum(memberships, axis=1, keepdims=True)
                memberships = np.nan_to_num(memberships, nan=1.0/self.n_clusters)

            # Step 3: Update centers
            new_centers = np.zeros_like(self.centers_)
            for i in range(self.n_clusters):
                weights = memberships[:,i]**self.m
                new_centers[i] = np.sum(X * weights[:,np.newaxis], axis=0) / np.sum(weights)

            # Step 4: Update covariances
            new_covariances = []
            for i in range(self.n_clusters):
                diff = X - new_centers[i]
                weights = memberships[:,i]**self.m

                # Compute weighted covariance
                weighted_cov = (diff.T * weights) @ diff / np.sum(weights)

                # Normalize to keep cluster volume constant
                det = np.linalg.det(weighted_cov)**(1/n_features)
                if det > 1e-10:
                    new_cov = weighted_cov / det
                else:
                    new_cov = weighted_cov

                new_covariances.append(new_cov)

            # Check convergence
            if np.allclose(new_centers, self.centers_, rtol=1e-4):
                break

            self.centers_ = new_centers
            self.covariances_ = new_covariances

        self.memberships_ = memberships
        return self

    def membership_degrees(self, x):
        """Calculate membership degrees for new point"""
        distances = np.zeros(self.n_clusters)
        for i in range(self.n_clusters):
            try:
                inv_cov = np.linalg.inv(self.covariances_[i])
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(self.covariances_[i])

            diff = x - self.centers_[i]
            distances[i] = np.sqrt(diff.T @ inv_cov @ diff)

        power = 2/(self.m-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            memberships = 1 / (distances ** power)
            memberships = memberships / np.sum(memberships)
            memberships = np.nan_to_num(memberships, nan=1.0/self.n_clusters)

        return memberships

class GGMembership:
    """Gath-Geva inspired membership functions"""

    def __init__(self, n_clusters=3, m=2, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.m = m  # Fuzziness coefficient
        self.max_iter = max_iter
        self.random_state = random_state
        self.centers_ = None
        self.covariances_ = None
        self.priors_ = None
        self.memberships_ = None

    def fit(self, X):
        """Learn cluster parameters using Gaussian Mixture Model"""
        n_samples, n_features = X.shape

        # Initialize with Gaussian Mixture Model
        gmm = GaussianMixture(n_components=self.n_clusters,
                            covariance_type='full',
                            max_iter=self.max_iter,
                            random_state=self.random_state)
        gmm.fit(X)

        self.centers_ = gmm.means_
        self.covariances_ = gmm.covariances_
        self.priors_ = gmm.weights_

        # Compute final memberships
        self.memberships_ = self._compute_memberships(X)
        return self

    def _compute_memberships(self, X):
        """Calculate membership degrees using Gaussian distribution"""
        n_samples = X.shape[0]
        memberships = np.zeros((n_samples, self.n_clusters))

        for i in range(self.n_clusters):
            # Compute probability density for each cluster
            diff = X - self.centers_[i]
            try:
                inv_cov = np.linalg.inv(self.covariances_[i])
                det_cov = np.linalg.det(self.covariances_[i])
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(self.covariances_[i])
                det_cov = np.linalg.det(self.covariances_[i] + np.eye(self.covariances_[i].shape[0])*1e-6)

            exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
            norm_factor = 1/((2*np.pi)**(X.shape[1]/2) * 1/np.sqrt(det_cov))
            prob = norm_factor * np.exp(exponent)

            memberships[:,i] = (self.priors_[i] * prob) ** (1/(self.m-1))

        # Normalize memberships
        memberships = memberships / np.sum(memberships, axis=1, keepdims=True)
        return memberships

    def membership_degrees(self, x):
        """Calculate membership degrees for new point"""
        x = np.array(x).reshape(1, -1)
        return self._compute_memberships(x)[0]

class FuzzyHarmonySearchFS:
    def __init__(self, num_agents=20, max_iter=100, HMCR=0.75, PAR=0.4,
                 weights=(0.6, 0.2, 0.2), random_state=42, verbose=True,
                 membership_type='gk', n_clusters=3):
        """
        Initialize the Fuzzy Harmony Search for Feature Selection.

        Args:
            membership_type: 'gk' for Gustafson-Kessel or 'gg' for Gath-Geva
            n_clusters: Number of clusters for membership functions
        """
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.HMCR = HMCR
        self.PAR = PAR
        self.weights = weights
        self.random_state = random_state
        self.verbose = verbose
        self.membership_type = membership_type
        self.n_clusters = n_clusters

        np.random.seed(random_state)
        random.seed(random_state)

        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.execution_time = None

        # Initialize membership functions
        if membership_type.lower() == 'gk':
            self.membership = GKMembreship(n_clusters=n_clusters, random_state=random_state)
        elif membership_type.lower() == 'gg':
            self.membership = GGMembership(n_clusters=n_clusters, random_state=random_state)
        else:
            raise ValueError("membership_type must be either 'gk' or 'gg'")

    def initialize_harmony_memory(self, num_features):
        """Initialize the harmony memory with random binary vectors."""
        return np.random.randint(0, 2, size=(self.num_agents, num_features))

    def calculate_feature_importance(self, X_train, y_train):
        """Calculate initial feature importance using mutual information."""
        return mutual_info_classif(X_train, y_train, random_state=self.random_state)

    def _initialize_membership(self, X_train):
        """Initialize membership functions with training data"""
        self.membership.fit(X_train)

    def _fuzzify_with_membership(self, values, value_range):
        """
        Fuzzify values using GK/GG membership degrees

        Args:
            values: Array of values to fuzzify
            value_range: Tuple of (min_value, max_value) for normalization
        """
        # Ensure values is numpy array
        values = np.asarray(values)
        
        # Handle scalar input
        if values.ndim == 0:
            values = values.reshape(1)
        
        # Normalize values to [0,1] range
        min_val, max_val = value_range
        if max_val == min_val:
            norm_values = np.ones_like(values) * 0.5  # Default to middle if no range
        else:
            norm_values = (values - min_val) / (max_val - min_val + 1e-10)
        
        norm_values = np.clip(norm_values, 0, 1)

        # Reshape for membership calculation
        if len(norm_values.shape) == 1:
            norm_values = norm_values.reshape(-1, 1)

        # Get membership degrees for each normalized value
        membership_degrees = np.array([self.membership.membership_degrees(x) for x in norm_values])

        # Handle single value case
        if membership_degrees.ndim == 1:
            return membership_degrees / np.sum(membership_degrees)
        
        # Aggregate membership degrees across clusters (take max across clusters)
        fuzzified = np.max(membership_degrees, axis=1)
        
        # Normalize if we have multiple values
        if len(fuzzified) > 1:
            total = np.sum(fuzzified)
            if total > 0:
                return fuzzified / total
            else:
                return np.ones_like(fuzzified) / len(fuzzified)
        else:
            return fuzzified

    def fuzzify_accuracy(self, accuracy):
        """Fuzzify accuracy using GK/GG membership"""
        result = self._fuzzify_with_membership(np.array([accuracy]), (0, 1))
        return result[0] if isinstance(result, np.ndarray) and len(result) > 0 else result

    def fuzzify_stability(self, stability):
        """Fuzzify stability using GK/GG membership"""
        result = self._fuzzify_with_membership(np.array([stability]), (0, 1))
        return result[0] if isinstance(result, np.ndarray) and len(result) > 0 else result

    def fuzzify_redundancy(self, redundancy):
        """Fuzzify redundancy using GK/GG membership"""
        result = self._fuzzify_with_membership(np.array([redundancy]), (0, 1))
        return result[0] if isinstance(result, np.ndarray) and len(result) > 0 else result

    def fuzzy_inference_system(self, accuracy_fuzzy, stability_fuzzy, redundancy_fuzzy):
        """
        Simplified inference system using GK/GG membership

        Here we use membership degrees directly instead of traditional fuzzy rules
        """
        # Handle scalar vs array inputs
        if hasattr(accuracy_fuzzy, '__len__') and len(accuracy_fuzzy) > 1:
            accuracy_val = np.max(accuracy_fuzzy)
        else:
            accuracy_val = float(accuracy_fuzzy)
            
        if hasattr(stability_fuzzy, '__len__') and len(stability_fuzzy) > 1:
            stability_val = np.max(stability_fuzzy)
        else:
            stability_val = float(stability_fuzzy)
            
        if hasattr(redundancy_fuzzy, '__len__') and len(redundancy_fuzzy) > 1:
            redundancy_val = np.max(redundancy_fuzzy)
        else:
            redundancy_val = float(redundancy_fuzzy)

        # Combine membership degrees using weighted product
        combined = (accuracy_val**self.weights[0] *
                   stability_val**self.weights[1] *
                   (1 - redundancy_val)**self.weights[2])

        # Normalize to get final fitness
        return {'combined': combined}

    def defuzzify_fitness(self, fitness_fuzzy):
        """Defuzzify using the combined value directly"""
        return fitness_fuzzy['combined']

    def fuzzy_stability_score(self, feature, X_train, y_train):
        """
        Compute stability score for a feature using fuzzy logic approach.
        """
        # Normalized variance-to-mean ratio
        feature_data = X_train[:, feature]
        mean_val = np.mean(feature_data)
        var_mean_ratio = np.var(feature_data) / (abs(mean_val) + 1e-10)
        stability_var = 1 / (1 + var_mean_ratio)

        # Entropy-based stability
        hist, _ = np.histogram(feature_data, bins=10, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        feature_entropy = entropy(hist)
        max_entropy = np.log(10)  # Maximum entropy for 10 bins
        stability_entropy = 1 - (feature_entropy / max_entropy) if max_entropy > 0 else 0

        # Combine stability measures
        combined_stability = 0.6 * stability_var + 0.4 * stability_entropy
        return combined_stability

    def fuzzy_redundancy_score(self, harmony, X_train):
        """
        Compute redundancy score considering correlation and feature diversity.
        """
        selected_features = np.where(harmony == 1)[0]
        num_selected = len(selected_features)

        if num_selected < 2:
            return 0.0  # No redundancy if only one or no features selected

        # Get selected feature data
        X_selected = X_train[:, selected_features]
        
        # Ensure we have a 2D array - this fixes the AxisError
        if X_selected.ndim == 1:
            X_selected = X_selected.reshape(-1, 1)
        
        # Handle single feature case after reshape
        if X_selected.shape[1] < 2:
            return 0.0

        try:
            # Correlation-based redundancy with proper transpose
            corr_matrix = np.corrcoef(X_selected.T)  # Transpose for features as rows
            
            # Handle scalar correlation (happens with 2 features)
            if corr_matrix.ndim == 0:
                return float(abs(corr_matrix))
            
            # For matrix case, get upper triangular part excluding diagonal
            np.fill_diagonal(corr_matrix, 0)
            abs_corr = np.abs(corr_matrix)
            
            # Get upper triangular values only
            upper_triangular = abs_corr[np.triu_indices_from(abs_corr, k=1)]
            
            redundancy_corr = np.mean(upper_triangular) if len(upper_triangular) > 0 else 0.0
            
        except Exception as e:
            # Fallback: compute pairwise correlations manually
            redundancy_corr = 0.0
            pair_count = 0
            
            for i in range(num_selected):
                for j in range(i + 1, num_selected):
                    try:
                        corr = np.corrcoef(X_selected[:, i], X_selected[:, j])[0, 1]
                        if not np.isnan(corr):
                            redundancy_corr += abs(corr)
                            pair_count += 1
                    except:
                        continue
            
            redundancy_corr = redundancy_corr / pair_count if pair_count > 0 else 0.0

        return redundancy_corr

    def compute_fuzzy_fitness(self, harmony, X_train, X_val, y_train, y_val):
        """
        Compute fitness using fuzzy inference system.
        """
        selected_features = np.where(harmony == 1)[0]
        num_selected = len(selected_features)

        if num_selected == 0:
            return 0.0

        # Get basic metrics
        X_train_subset = X_train[:, selected_features]
        X_val_subset = X_val[:, selected_features]

        # Ensure 2D arrays
        if X_train_subset.ndim == 1:
            X_train_subset = X_train_subset.reshape(-1, 1)
        if X_val_subset.ndim == 1:
            X_val_subset = X_val_subset.reshape(-1, 1)

        classifier = RandomForestClassifier(
            n_estimators=50,  # Reduced for faster computation
            class_weight='balanced',
            random_state=self.random_state
        )
        classifier.fit(X_train_subset, y_train)
        predictions = classifier.predict(X_val_subset)

        # Calculate crisp values
        accuracy = accuracy_score(y_val, predictions)
        stability_scores = np.mean([self.fuzzy_stability_score(f, X_train, y_train)
                                  for f in selected_features])
        redundancy_score = self.fuzzy_redundancy_score(harmony, X_train)

        # Fuzzify inputs
        accuracy_fuzzy = self.fuzzify_accuracy(accuracy)
        stability_fuzzy = self.fuzzify_stability(stability_scores)
        redundancy_fuzzy = self.fuzzify_redundancy(redundancy_score)

        # Apply fuzzy inference system
        fitness_fuzzy = self.fuzzy_inference_system(accuracy_fuzzy, stability_fuzzy, redundancy_fuzzy)

        # Defuzzify to get final fitness
        fuzzy_fitness = self.defuzzify_fitness(fitness_fuzzy)

        # Apply size penalty
        size_penalty = 1 - (num_selected / X_train.shape[1]) * 0.1
        final_fitness = fuzzy_fitness * size_penalty

        return final_fitness

    def sort_agents(self, harmony_memory, fitness):
        """Sort the harmony memory based on fitness values."""
        sorted_indices = np.argsort(fitness)[::-1]
        return harmony_memory[sorted_indices], fitness[sorted_indices]

    def fit(self, X, y):
        """
        Perform feature selection using Fuzzy Harmony Search.
        """
        start_time = time.time()
        num_features = X.shape[1]

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state, stratify=y
        )

        # Initialize membership functions with training data
        self._initialize_membership(X_train)

        # Calculate feature importance
        self.feature_importance = self.calculate_feature_importance(X_train, y_train)

        # Initialize harmony memory
        harmony_memory = self.initialize_harmony_memory(num_features)
        fitness = np.zeros(self.num_agents)

        # Compute initial fitness using fuzzy logic
        for i in range(self.num_agents):
            fitness[i] = self.compute_fuzzy_fitness(
                harmony_memory[i], X_train, X_val, y_train, y_val
            )

        # Sort harmony memory
        harmony_memory, fitness = self.sort_agents(harmony_memory, fitness)

        # Initialize best solution
        self.best_harmony = harmony_memory[0]
        self.best_fitness = fitness[0]

        # Track fitness history
        self.best_fitness_history.append(self.best_fitness)
        self.avg_fitness_history.append(np.mean(fitness))

        # Main optimization loop
        iterator = range(self.max_iter)
        if self.verbose:
            iterator = tqdm(iterator, desc="Fuzzy Harmony Search Progress")

        for iteration in iterator:
            # More aggressive dynamic parameters for better exploration
            progress = iteration / self.max_iter
            current_HMCR = self.HMCR * (0.5 + 0.5 * np.cos(np.pi * progress))  # Oscillating HMCR
            current_PAR = self.PAR * (0.5 + 1.5 * progress)  # More aggressive PAR increase

            # Create new harmony
            new_harmony = np.zeros(num_features, dtype=int)

            for feature_index in range(num_features):
                rand_val = np.random.rand()

                if rand_val < current_HMCR:
                    # Enhanced harmony memory consideration with tournament selection
                    if rand_val < current_HMCR * 0.7:  # 70% of HMCR uses fitness-based selection
                        exp_fitness = np.exp(fitness * 2)
                        sum_exp_fitness = np.sum(exp_fitness)
                        if (np.any(np.isnan(fitness)) or np.any(np.isinf(fitness)) or sum_exp_fitness == 0 or np.isnan(sum_exp_fitness)):
                            print("[FHS WARNING] Invalid fitness values detected. Using uniform probabilities.")
                            probabilities = np.ones(self.num_agents) / self.num_agents
                        else:
                            probabilities = exp_fitness / sum_exp_fitness
                        random_agent = np.random.choice(self.num_agents, p=probabilities)
                    else:  # 30% uses tournament selection
                        tournament_size = min(3, self.num_agents)
                        tournament_indices = np.random.choice(self.num_agents, tournament_size, replace=False)
                        tournament_fitness = fitness[tournament_indices]
                        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                        random_agent = winner_idx

                    new_harmony[feature_index] = harmony_memory[random_agent, feature_index]

                    # Enhanced pitch adjustment with adaptive probability
                    adaptive_PAR = current_PAR * (1 + 0.5 * np.random.normal(0, 0.1))  # Add noise
                    adaptive_PAR = np.clip(adaptive_PAR, 0, 1)

                    if np.random.rand() < adaptive_PAR:
                        new_harmony[feature_index] = 1 - new_harmony[feature_index]
                else:
                    # Improved random generation with importance and diversity consideration
                    if hasattr(self, 'feature_importance'):
                        # Balance between importance and diversity
                        importance_prob = self.feature_importance[feature_index] / (np.max(self.feature_importance) + 1e-10)

                        # Add diversity factor - prefer less selected features
                        selection_frequency = np.mean(harmony_memory[:, feature_index])
                        diversity_factor = 1 - selection_frequency

                        combined_prob = 0.1 + 0.6 * importance_prob + 0.3 * diversity_factor
                        new_harmony[feature_index] = 1 if np.random.rand() < combined_prob else 0
                    else:
                        new_harmony[feature_index] = np.random.randint(0, 2)

            # Ensure reasonable number of features (between 1 and 70% of total)
            max_features = max(1, int(0.7 * num_features))
            if np.sum(new_harmony) == 0:
                # Select top important features
                if hasattr(self, 'feature_importance'):
                    top_features = np.argsort(self.feature_importance)[-3:]  # Select top 3
                    new_harmony[top_features] = 1
                else:
                    new_harmony[random.randint(0, num_features - 1)] = 1
            elif np.sum(new_harmony) > max_features:
                # Randomly remove excess features, but keep important ones
                selected_indices = np.where(new_harmony == 1)[0]
                if hasattr(self, 'feature_importance'):
                    # Sort by importance and keep the most important
                    importance_of_selected = self.feature_importance[selected_indices]
                    sorted_indices = selected_indices[np.argsort(importance_of_selected)]
                    to_remove = sorted_indices[:len(selected_indices) - max_features]
                else:
                    to_remove = np.random.choice(selected_indices,
                                               len(selected_indices) - max_features,
                                               replace=False)
                new_harmony[to_remove] = 0

            # Compute fuzzy fitness
            new_fitness = self.compute_fuzzy_fitness(
                new_harmony, X_train, X_val, y_train, y_val
            )

            # Update harmony memory with better replacement strategy
            worst_idx = np.argmin(fitness)
            if new_fitness > fitness[worst_idx]:
                harmony_memory[worst_idx] = new_harmony
                fitness[worst_idx] = new_fitness

                # Sort harmony memory again
                harmony_memory, fitness = self.sort_agents(harmony_memory, fitness)

                # Update the best harmony if improved
                if fitness[0] > self.best_fitness:
                    self.best_harmony = harmony_memory[0].copy()
                    self.best_fitness = fitness[0]
            else:
                # Even if not better than worst, occasionally accept with small probability
                if np.random.rand() < 0.05:  # 5% chance to accept worse solution
                    random_replace = np.random.randint(self.num_agents // 2, self.num_agents)
                    harmony_memory[random_replace] = new_harmony
                    fitness[random_replace] = new_fitness

            # Record fitness history
            self.best_fitness_history.append(self.best_fitness)
            self.avg_fitness_history.append(np.mean(fitness))

            # Early stopping
            if iteration > 0.1 * self.max_iter:
                recent_improvement = max(self.best_fitness_history[-int(0.1*self.max_iter):])
                if recent_improvement <= self.best_fitness_history[-int(0.1*self.max_iter)-1]:
                    if self.verbose:
                        print(f"\nEarly stopping at iteration {iteration} - no improvement")
                    break

        self.execution_time = time.time() - start_time

        if self.verbose:
            print("\nFuzzy Harmony Search feature selection completed.")
            print(f"Execution time: {self.execution_time:.2f} seconds")
            print(f"Selected {np.sum(self.best_harmony)}/{num_features} features: "
                  f"{np.where(self.best_harmony == 1)[0]}")

        return self

    def transform(self, X):
        """Return the dataset with selected features."""
        if not hasattr(self, 'best_harmony'):
            raise RuntimeError("The model must be fitted before transformation.")
        return X[:, self.best_harmony == 1]

    def fit_transform(self, X, y):
        """Fit the model and return the transformed dataset."""
        self.fit(X, y)
        return self.transform(X)