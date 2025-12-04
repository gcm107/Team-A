
# model loading and prediction functions for stock price prediction.


import os
import json
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression


# model file paths
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.joblib")
REGRESSOR_PATH = os.path.join(MODEL_DIR, "regressor.joblib")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.json")

# global variables to cache loaded models
_scaler = None
_classifier = None
_regressor = None
_feature_cols = None


def load_scaler() -> StandardScaler:
    """function to load and return the StandardScaler model."""
    global _scaler
    if _scaler is None:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler model not found at {SCALER_PATH}")
        _scaler = joblib.load(SCALER_PATH)
    return _scaler


def load_classifier() -> LogisticRegression:
    """function to load and return the LogisticRegression classifier."""
    global _classifier
    if _classifier is None:
        if not os.path.exists(CLASSIFIER_PATH):
            raise FileNotFoundError(f"Classifier model not found at {CLASSIFIER_PATH}")
        _classifier = joblib.load(CLASSIFIER_PATH)
    return _classifier


def load_regressor() -> LinearRegression:
    """function to load and return the LinearRegression regressor."""
    global _regressor
    if _regressor is None:
        if not os.path.exists(REGRESSOR_PATH):
            raise FileNotFoundError(f"Regressor model not found at {REGRESSOR_PATH}")
        _regressor = joblib.load(REGRESSOR_PATH)
    return _regressor


def get_feature_cols() -> list:
    """function to load and return the feature column names."""
    global _feature_cols
    if _feature_cols is None:
        if not os.path.exists(FEATURE_COLS_PATH):
            raise FileNotFoundError(f"Feature columns file not found at {FEATURE_COLS_PATH}")
        with open(FEATURE_COLS_PATH, "r") as f:
            _feature_cols = json.load(f)
    return _feature_cols


def predict_up_down(daily_return: float, mean_5d: float, vol_5d: float, spy_return: float) -> dict:
    """
    func to predict whether stock will go up (1) or down (0) in the next 5 days.
    """
    # load models
    scaler = load_scaler()
    classifier = load_classifier()
    
    # prepare input as numpy array
    features = np.array([[daily_return, mean_5d, vol_5d, spy_return]])
    
    # scale features
    features_scaled = scaler.transform(features)
    
    # predict
    prediction = classifier.predict(features_scaled)[0]
    probability = classifier.predict_proba(features_scaled)[0]
    
    return {
        "prediction": int(prediction),
        "probability": float(probability[1])  # probability of class 1 (up)
    }


def predict_return(daily_return: float, mean_5d: float, vol_5d: float, spy_return: float) -> dict:
    # function to predict forward return
    # load models
    scaler = load_scaler()
    regressor = load_regressor()
    
    # prepare input as numpy array
    features = np.array([[daily_return, mean_5d, vol_5d, spy_return]])
    
    # scale features
    features_scaled = scaler.transform(features)
    
    # predict
    prediction = regressor.predict(features_scaled)[0]
    
    return {
        "prediction": float(prediction)
    }

