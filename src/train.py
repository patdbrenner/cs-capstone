from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from src.explore import DATA_PATH, LABEL_COLUMN, TEXT_COLUMN, load_data
from src.preprocess import clean_series

logger = logging.getLogger(__name__)

MODEL_DIR = Path('models')
MODEL_PATH = MODEL_DIR / 'model.pkl'
VECTORIZER_PATH = MODEL_DIR / 'vectorizer.pkl'
METADATA_PATH = MODEL_DIR / 'metadata.json'

RANDOM_STATE = 964
TEST_SIZE = 0.2

PARAM_DISTRIBUTIONS = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(10, 100),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2'],}

N_ITER_SEARCH = 20
CV_FOLDS = 3

def prepare_features(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    '''
    Clean ticket text and return features and labels.
    
    Args:
        df: Raw ticket DataFrame with TEXT_COLUMN and LABEL_COLUMN.
        
    Returns:
        Tuple of (cleaned text series, label series).
    '''
    logger.info('Preparing features from raw data')
    cleaned_text = clean_series(df[TEXT_COLUMN])
    labels = df[LABEL_COLUMN]
    return cleaned_text, labels

def vectorize_text(X_train: pd.Series, X_test: pd.Series) -> tuple[TfidfVectorizer, csr_matrix, csr_matrix]:
    '''
    Fit a TF_IDF vectorizer on training text and transform both splits.
    
    Args:
        X_train: Cleaned training text.
        X_test: Cleaned test text.
    
    Returns:
        Tuple of (fitted vectorizer, train feature matrix, test feature matrix).
    '''
    logger.info('Fitting TF-IDF vectorizer on training data')
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    logger.info(f'TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}')
    return vectorizer, X_train_vec, X_test_vec

def tune_random_forest(X_train_vec, y_train: pd.Series) -> RandomForestClassifier:
    '''
    Run RandomizedSearchCV to find good Random Forest hyperparameters.
    
    Args:
        x_train_vec: TF-IDF feature matrix for training data.
        y_train: Training labels.
        
    Returns:
        Best estimator found by the search, already fit on training data.
    '''
    logger.info(f'Starting RandomizedSearchCV with {N_ITER_SEARCH} iterations and {CV_FOLDS} folds')
    base_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,)
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER_SEARCH,
        cv=CV_FOLDS,
        scoring="f1_weighted",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=2,)
    search.fit(X_train_vec, y_train)
    logger.info(f'Best params found: {search.best_params_}')
    logger.info(f'Best cross-validated weighted F1: {search.best_score_:.4f}')
    return search.best_estimator_, search.best_params_, search.best_score_

def save_artifacts(model: RandomForestClassifier, vectorizer: TfidfVectorizer) -> None:
    '''
    Persist the trained model and vectorizer to local storage.
    
    Args:
        model: Trained Random Forest classifier.
        vectorizer: Fitted TF-IDF vectorizer.
    '''
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    logger.info(f'Saved model to {MODEL_PATH}')
    logger.info(f'Saved vectorizer to {VECTORIZER_PATH}')

def save_metadata(
    best_params: dict, cv_f1: float, test_f1: float, train_size: int, test_size: int
) -> None:
    '''
    Persist training metadata to storage for dashboard display.
    
    Args:
        best_params: Best hyperparameters found by RandomizedSearchCV.
        cv_f1: Best cross-validated weighted F1 score.
        test_f1: Weighted F1 score on the test set.
        train_size: Number of training examples.
        test_size: Number of test examples.
    '''

    metadata = {
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'best_params': best_params,
        'cv_weighted_f1': cv_f1,
        'test_weighted_f1': test_f1,
        'train_size': train_size,
        'test_size': test_size,}
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f'Saved training metadata to {METADATA_PATH}')

def run_training(path: str = DATA_PATH) -> None:
    '''
    Run the full training pipeline: load, prepare, split, vectorize, tune, save.
    
    Args:
        path: Path to the raw CSV dataset.
    '''
    df = load_data(path)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,)
    logger.info(f'Train size: {len(X_train)}, Test size: {len(X_test)}')

    vectorizer, X_train_vec, X_test_vec = vectorize_text(X_train, X_test)

    best_model, best_params, cv_f1 = tune_random_forest(X_train_vec, y_train)

    test_predictions = best_model.predict(X_test_vec)
    test_f1 = f1_score(y_test, test_predictions, average='weighted')
    logger.info(f'Test set weighted F1: {test_f1:.4f}')

    save_artifacts(best_model, vectorizer)
    save_metadata(best_params, cv_f1, test_f1, len(X_train), len(X_test))

    joblib.dump((X_test_vec, y_test), MODEL_DIR / "test_split.pkl")
    logger.info("Saved test split for eval")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    run_training()