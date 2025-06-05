"""
breast_cancer_rf_eval.py

A script to evaluate Random Forest model performance on the Breast Cancer dataset
using both original and log-transformed feature sets.

Author: [Your Name]
License: MIT
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def evaluate_random_forest(X, y, apply_log=False, test_size=0.3, random_state=42, verbose=True):
    """
    Trains and evaluates a Random Forest model.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target labels.
        apply_log (bool): Apply log1p transformation to features.
        test_size (float): Fraction of data for testing.
        random_state (int): Seed for reproducibility.
        verbose (bool): Print results.

    Returns:
        dict: Model, predictions, and accuracy.
    """
    if apply_log:
        X = X.apply(np.log1p)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if verbose:
        print(f"\n{'Log-' if apply_log else ''}Transformed Features:")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

    return {
        "model": model,
        "predictions": y_pred,
        "accuracy": accuracy
    }


def main():
    """
    Main function to run model evaluations on original and log-transformed data.
    """
    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    print("🔍 Evaluating Random Forest Classifier on Breast Cancer Dataset")

    # Evaluate on original features
    result_orig = evaluate_random_forest(X, y, apply_log=False)

    # Evaluate on log-transformed features
    result_log = evaluate_random_forest(X, y, apply_log=True)


if __name__ == "__main__":
    main()
