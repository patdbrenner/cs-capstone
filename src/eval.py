from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score

from src.visualizations import (build_confusion_matrix_figure,
                                build_feature_importance_figure)

logger = logging.getLogger(__name__)

MODEL_DIR = Path('models')
MODEL_PATH = MODEL_DIR / 'model.pkl'
VECTORIZER_PATH = MODEL_DIR / 'vectorizer.pkl'
TEST_SPLIT_PATH = MODEL_DIR / 'test_split.pkl'

OUTPUT_DIR = Path('outputs')
CONFUSION_MATRIX_PATH = OUTPUT_DIR / 'confusion_matrix.png'
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / 'feature_importance.png'

TOP_N_FEATURES = 20

def load_artifacts() -> tuple[RandomForestClassifier, TfidfVectorizer]:
    '''
    Load the trained model and fitted vectorizer from storage.
    
    Returns:
        Tuple of (trained model, fitted vectorizer).
    '''
    logger.info(f'Loading model from {MODEL_PATH}')
    model = joblib.load(MODEL_PATH)
    logger.info(f'Loading vectorizer from {VECTORIZER_PATH}')
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def load_test_split() -> tuple:
    '''
    Load the test set saved during training.
    
    Returns:
        Tuple of (test feature matrix, test labels).
    '''
    logger.info(f'Loading test split from {TEST_SPLIT_PATH}')
    X_test_vec, y_test = joblib.load(TEST_SPLIT_PATH)
    return X_test_vec, y_test

def evaluate_model(model: RandomForestClassifier, X_test_vec, y_test) -> dict:
    '''
    Generate predictions and compute weighted F1 and full classification report.
    
    Args:
        model: Trained Random Forest classifier.
        X_test_vec: TF-IDF feature matrix for the test set.
        y_test: True labels for the test set.
        
    Returns:
        Dictionary containing predictions, weighted F1, and classification report.
    '''
    logger.info('Generating predictions on test set')
    y_pred = model.predict(X_test_vec)

    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    logger.info(f'Test set weighted F1: {weighted_f1:.4f}')

    report = classification_report(y_test, y_pred)
    logger.info(f'Classification report:\n{report}')

    return {
        'y_pred': y_pred,
        'weighted_f1': weighted_f1,
        'report': report,}

def save_confusion_matrix(y_test, y_pred, labels: list[str]) -> None:
    '''
    Build a confusion matrix figure and save it to storage.
    
    Args:
        y_test: True labels for the test set.
        y_pred: Predicted labels for the test set.
        labels: Ordered list of class labels for axis ticks.
    '''
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig = build_confusion_matrix_figure(y_test, y_pred, labels=labels)
    fig.savefig(CONFUSION_MATRIX_PATH)
    plt.close(fig)
    logger.info(f'Saved confusion matrix to {CONFUSION_MATRIX_PATH}')

def save_feature_importance(
        model: RandomForestClassifier, vectorizer: TfidfVectorizer, top_n: int = TOP_N_FEATURES
) -> None:
    '''
    Build a feature importance figure and save it to storage.
    
    Args:
        model: Trained Random Forest classifier.
        vectorizer: Fitted TF-IDF vectorizer, used to map feature indices to terms.
        top_n: Number of top features to display.
    '''
    OUTPUT_DIR.mkdir(exist_ok=True)
    feature_names = np.array(vectorizer.get_feature_names_out())
    importances = model.feature_importances_

    fig = build_feature_importance_figure(feature_names, importances, top_n)
    fig.savefig(FEATURE_IMPORTANCE_PATH)
    plt.close(fig)
    logger.info(f'Saved feature importance chart to {FEATURE_IMPORTANCE_PATH}')

def run_eval() -> None:
    '''
    Run evaluation pipeline: load artifacts, evaluate, generate plots.
    '''
    model, vectorizer = load_artifacts()
    X_test_vec, y_test = load_test_split()

    results = evaluate_model(model, X_test_vec, y_test)

    labels = sorted(y_test.unique())
    save_confusion_matrix(y_test, results['y_pred'], labels)
    save_feature_importance(model, vectorizer)

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    run_eval()