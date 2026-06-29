from __future__ import annotations

import json
from pathlib import Path

import joblib
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_PATH = 'models/model.pkl'
VECTORIZER_PATH = 'models/vectorizer.pkl'
TEST_SPLIT_PATH = 'models/test_split.pkl'
METADATA_PATH = 'models/metadata.json'

@st.cache_resource
def load_artifacts() -> tuple[RandomForestClassifier, TfidfVectorizer]:
    '''
    Load the trained model and fitted vectorizer, cached across reruns.
    
    Returns:
        Tuple of (trained model, fitted vectorizer).
    '''
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

@st.cache_resource
def load_test_split() -> tuple:
    '''
    Load the test set used to generate dashboard visualizations.
    
    Returns:
        Tuple of (test feature matrix, test labels).
    '''
    X_test_vec, y_test = joblib.load(TEST_SPLIT_PATH)
    return X_test_vec, y_test

@st.cache_data
def load_metadata() -> dict:
    '''
    Load training metadata for display on the dashboard.
    
    Returns:
        Dictionary of training metadata, empty if no metadata is present.
    '''
    if not Path(METADATA_PATH).exists():
        return {}
    
    with open(METADATA_PATH, 'r') as f:
        return json.load(f)