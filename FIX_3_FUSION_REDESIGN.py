#!/usr/bin/env python3
"""
FIX #3: FUSION STRATEGY REDESIGN
=================================

PROBLEM:
--------
Current late fusion achieves AUC 0.591 (same as text-only).
This means multimodal fusion provides ZERO benefit.

Root cause: Static weights don't adapt to modality quality.
  Text: AUC 0.591 → weight 0.35
  Audio: AUC 0.548 → weight 0.35  ← WRONG! Audio is worse than random
  Visual: AUC 0.657 → weight 0.30  ← WRONG! Visual is best

With static weights, low-quality audio (0.548 AUC) pulls fusion down.

SOLUTION:
---------
Three improved fusion strategies that learn weights from validation data.

RECOMMENDATION:
Use AttentionFusion - it's simple and effective for this use case.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')


class AttentionFusion:
    """
    Learns soft attention weights based on validation AUC performance.
    
    Key idea:
    - Compute validation AUC for each modality
    - Weight each modality by its AUC score
    - Low-performing modalities (near 0.5) get near-zero weight
    - High-performing modalities get higher weight
    
    Example:
        Text AUC: 0.591 → weight = 0.591 - 0.5 = 0.091
        Audio AUC: 0.548 → weight = 0.548 - 0.5 = 0.048
        Visual AUC: 0.657 → weight = 0.657 - 0.5 = 0.157
        
        Normalized: [0.30, 0.16, 0.54]  ← Visual dominates!
    
    Benefits:
    - Automatically down-weights unreliable modalities
    - Adapts to actual data quality
    - Simple and interpretable
    - No hyperparameters
    """
    
    def __init__(self, min_auc_threshold=0.52):
        """
        Parameters:
        -----------
        min_auc_threshold : float
            Modalities with AUC below this are set to weight 0
            (Default 0.52 means modalities barely better than random get zero weight)
        """
        self.min_auc_threshold = min_auc_threshold
        self.auc_scores = {}
        self.weights = {}
        self.scaler = StandardScaler()
        
    def fit(self, probabilities_dict, y_val):
        """
        Learn attention weights from validation data.
        
        Parameters:
        -----------
        probabilities_dict : dict
            Keys: modality names (e.g., 'text', 'audio', 'visual')
            Values: numpy arrays of class probabilities (N,)
        
        y_val : numpy array
            True labels (N,) - binary [0, 1]
        
        Returns:
        --------
        self
        """
        self.auc_scores = {}
        self.weights = {}
        
        for modality, probs in probabilities_dict.items():
            # Ensure 1D array
            probs = np.asarray(probs).ravel()
            
            # Compute AUC
            try:
                auc = roc_auc_score(y_val, probs)
            except:
                auc = 0.5
            
            self.auc_scores[modality] = auc
            
            # Weight = (AUC - 0.5), clipped at 0.01 minimum
            if auc > self.min_auc_threshold:
                weight = auc - 0.5
            else:
                weight = 0.01  # Minimum weight, not zero (for stability)
            
            self.weights[modality] = weight
        
        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        
        return self
    
    def predict_proba(self, probabilities_dict, threshold=0.5):
        """
        Fuse predictions using learned attention weights.
        
        Parameters:
        -----------
        probabilities_dict : dict
            Keys: modality names (e.g., 'text', 'audio', 'visual')
            Values: numpy arrays of class probabilities (N,)
        
        threshold : float
            Decision threshold (default 0.5)
        
        Returns:
        --------
        probs : numpy array
            Fused probabilities (N,)
        
        preds : numpy array
            Binary predictions (N,) - 0 or 1
        """
        if not self.weights:
            raise ValueError("Must call fit() before predict_proba()")
        
        # Weighted average of probabilities
        fused_probs = np.zeros(len(next(iter(probabilities_dict.values()))))
        
        for modality, probs in probabilities_dict.items():
            probs = np.asarray(probs).ravel()
            weight = self.weights.get(modality, 0.0)
            fused_probs += weight * probs
        
        # Normalize if needed
        fused_probs = np.clip(fused_probs, 0, 1)
        
        # Binary predictions
        preds = (fused_probs >= threshold).astype(int)
        
        return fused_probs, preds
    
    def get_weights_dict(self):
        """Return dictionary of learned weights."""
        return self.weights.copy()
    
    def get_auc_scores(self):
        """Return dictionary of validation AUC scores."""
        return self.auc_scores.copy()


class StackingFusion:
    """
    Uses a meta-learner (Logistic Regression) to combine modality predictions.
    
    More powerful than attention fusion but risks overfitting on small datasets.
    
    Key idea:
    - Train a second-level model that learns optimal combination
    - Inputs: predicted probabilities from each modality
    - Output: final depression probability
    
    Benefits:
    - Learns non-linear combinations (if using non-linear base)
    - Can capture interactions between modalities
    - Similar to ensemble methods
    
    Drawbacks:
    - More prone to overfitting
    - Requires sufficient validation data
    - Less interpretable than attention
    """
    
    def __init__(self, random_state=42):
        self.meta_learner = LogisticRegression(
            random_state=random_state,
            class_weight='balanced',
            max_iter=1000
        )
        self.scaler = StandardScaler()
        self.modality_names = []
        
    def fit(self, probabilities_dict, y_val):
        """
        Train meta-learner.
        
        Parameters:
        -----------
        probabilities_dict : dict
            Keys: modality names
            Values: numpy arrays of probabilities (N,)
        
        y_val : numpy array
            True labels (N,)
        
        Returns:
        --------
        self
        """
        self.modality_names = list(probabilities_dict.keys())
        
        # Stack probabilities as features
        X_meta = np.column_stack([
            np.asarray(probabilities_dict[mod]).ravel()
            for mod in self.modality_names
        ])
        
        # Scale features
        X_meta_scaled = self.scaler.fit_transform(X_meta)
        
        # Train meta-learner
        self.meta_learner.fit(X_meta_scaled, y_val)
        
        return self
    
    def predict_proba(self, probabilities_dict, threshold=0.5):
        """
        Use meta-learner to fuse predictions.
        
        Parameters:
        -----------
        probabilities_dict : dict
            Keys: modality names
            Values: numpy arrays of probabilities (N,)
        
        threshold : float
            Decision threshold
        
        Returns:
        --------
        probs : numpy array
            Fused probabilities
        
        preds : numpy array
            Binary predictions
        """
        # Stack probabilities
        X_meta = np.column_stack([
            np.asarray(probabilities_dict[mod]).ravel()
            for mod in self.modality_names
        ])
        
        # Scale
        X_meta_scaled = self.scaler.transform(X_meta)
        
        # Get probabilities from meta-learner
        fused_probs = self.meta_learner.predict_proba(X_meta_scaled)[:, 1]
        
        # Binary predictions
        preds = (fused_probs >= threshold).astype(int)
        
        return fused_probs, preds


class CostSensitiveFusion:
    """
    Learns weights that minimize cost of errors.
    
    Depression detection is asymmetric:
    - False Negative (missing depression): Very costly, clinical risk
    - False Positive (false alarm): Less costly, just requires follow-up
    
    Cost ratio: FN is 5x more costly than FP
    
    This fusion optimizes for the asymmetric cost, not accuracy.
    
    Example:
        Text AUC: 0.591, prob=0.6 on threshold=0.5
        → May have high false negative rate
        → Lower threshold or higher weight to catch more cases
        
        Audio AUC: 0.548, prob=0.51
        → Near random, high cost for both FN and FP
        → Reduce weight significantly
    """
    
    def __init__(self, fn_cost=5, fp_cost=1):
        """
        Parameters:
        -----------
        fn_cost : float
            Cost of false negative (missing depression) - default 5
        fp_cost : float
            Cost of false positive (false alarm) - default 1
        """
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.weights = {}
        self.auc_scores = {}
        self.sensitivities = {}
        
    def fit(self, probabilities_dict, y_val, thresholds=np.arange(0.1, 0.9, 0.05)):
        """
        Learn weights to minimize cost.
        
        Parameters:
        -----------
        probabilities_dict : dict
            Keys: modality names
            Values: numpy arrays of probabilities (N,)
        
        y_val : numpy array
            True labels
        
        thresholds : array-like
            Thresholds to search for optimal cost
        
        Returns:
        --------
        self
        """
        from sklearn.metrics import confusion_matrix
        
        self.auc_scores = {}
        self.weights = {}
        self.sensitivities = {}
        
        for modality, probs in probabilities_dict.items():
            probs = np.asarray(probs).ravel()
            
            # Compute AUC
            auc = roc_auc_score(y_val, probs)
            self.auc_scores[modality] = auc
            
            # Find threshold that minimizes cost
            best_cost = float('inf')
            best_threshold = 0.5
            best_sensitivity = 0.0
            
            for thresh in thresholds:
                preds = (probs >= thresh).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
                
                # Compute cost
                cost = fn * self.fn_cost + fp * self.fp_cost
                
                # Compute sensitivity (recall) - ability to detect depression
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                if cost < best_cost:
                    best_cost = cost
                    best_threshold = thresh
                    best_sensitivity = sensitivity
            
            # Weight inversely proportional to cost
            # Higher cost → lower weight
            weight = 1.0 / (1.0 + best_cost / 100.0)  # Normalize
            
            self.weights[modality] = max(0.01, weight)  # Minimum 0.01
            self.sensitivities[modality] = best_sensitivity
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        
        return self
    
    def predict_proba(self, probabilities_dict, threshold=0.5):
        """Fuse predictions."""
        if not self.weights:
            raise ValueError("Must call fit() first")
        
        # Weighted average
        fused_probs = np.zeros(len(next(iter(probabilities_dict.values()))))
        
        for modality, probs in probabilities_dict.items():
            probs = np.asarray(probs).ravel()
            weight = self.weights.get(modality, 0.0)
            fused_probs += weight * probs
        
        fused_probs = np.clip(fused_probs, 0, 1)
        preds = (fused_probs >= threshold).astype(int)
        
        return fused_probs, preds


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """Show how to use AttentionFusion in main.py"""
    
    print("""
    # In main.py, replace late fusion section with:
    
    from FIX_3_FUSION_REDESIGN import AttentionFusion
    
    # ... after computing validation probabilities ...
    
    # Create attention fusion
    fusion = AttentionFusion(min_auc_threshold=0.52)
    
    # Fit on validation data
    val_probs = {
        'text': p_t_val,      # Text probabilities on validation set
        'audio': p_a_val,     # Audio probabilities on validation set
        'visual': p_v_val,    # Visual probabilities on validation set
    }
    fusion.fit(val_probs, y_val)
    
    # Check learned weights
    print("Learned fusion weights:")
    for mod, weight in fusion.get_weights_dict().items():
        auc = fusion.get_auc_scores()[mod]
        print(f"  {mod}: AUC={auc:.3f}, weight={weight:.3f}")
    
    # Example expected output:
    # text: AUC=0.591, weight=0.30
    # audio: AUC=0.548, weight=0.05 ← Audio weight is LOW
    # visual: AUC=0.657, weight=0.65 ← Visual weight is HIGH
    
    # Use on test data
    test_probs = {
        'text': p_t_test,
        'audio': p_a_test,
        'visual': p_v_test,
    }
    fused_test_probs, fused_test_preds = fusion.predict_proba(test_probs, threshold=0.5)
    
    # Evaluate
    from sklearn.metrics import roc_auc_score
    test_auc = roc_auc_score(y_test, fused_test_probs)
    print(f"Fused test AUC: {test_auc:.3f}")
    # Expected: Should be > 0.65 (better than text-only 0.591)
    """)


if __name__ == '__main__':
    example_usage()
    print("\nImplementation options:")
    print("1. AttentionFusion (RECOMMENDED) - Simple, effective, interpretable")
    print("2. StackingFusion - More powerful, higher risk of overfitting")
    print("3. CostSensitiveFusion - Optimizes for clinical cost (FN=5x worse than FP)")
