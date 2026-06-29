from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from app.prediction_log import load_prediction_log
from src.visualizations import (build_confusion_matrix_figure,
                                build_distribution_figure,
                                build_feature_importance_figure)

TOP_N_FEATURES = 20
RECENT_PRED_DISPLAY_COUNT = 10

def render_model_info(metadata: dict) -> None:
    '''
    Render a summary of model training metadata.
    
    Args:
        metadata: Dictionary of training metadata loaded from metadata.json.
    '''
    st.subheader('Model Information')

    if not metadata:
        st.info('No training metadata found.')
        return
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Tested Weighted F1', f'{metadata["test_weighted_f1"]:.2%}')
    col2.metric('CV Weighted F1', f'{metadata["cv_weighted_f1"]:.2%}')
    col3.metric('Training Examples', f'{metadata["train_size"]:,}')
    col4.metric('Test Examples', f'{metadata["test_size"]:,}')

    trained_at = metadata['trained_at'].split('T')[0]
    st.caption(f'Model last trained on {trained_at}')

    with st.expander('Best Hyperparameters'):
        st.json(metadata['best_params'])

def render_recent_predictions() -> None:
    '''
    Render a table of the most recent predictions logged through the app.
    '''
    st.subheader('Recent Predictions')

    log_df = load_prediction_log()
    if log_df.empty:
        st.info('No predictions logged yet. Submit a ticket to see it appear here.')
        return

    display_df = log_df.head(RECENT_PRED_DISPLAY_COUNT)[
        ['timestamp', 'title', 'predicted_category', 'confidence']]
    
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f'{x:.1%}')
    st.dataframe(display_df, width='stretch')

def render_distribution_chart(y_test) -> None:
    '''
    Render a bar chart of class distribution in the test set.
    
    Args:
        y_test: True labels for the test set.
    '''
    st.subheader('Category Distribution')
    counts = y_test.value_counts()
    fig = build_distribution_figure(counts)
    st.pyplot(fig)
    plt.close(fig)

def render_confusion_matrix(model: RandomForestClassifier, X_test_vec, y_test) -> None:
    '''
    Render a confusion matrix heatmap comparing predictions to true labels.
    
    Args:
        model: Trained Random Forest classifier.
        X_test_vec: TF-IDF feature matrix for the test set.
        y_test: True labels for the test set.
    '''
    st.subheader('Confusion Matrix')
    y_pred = model.predict(X_test_vec)
    labels = sorted(y_test.unique())
    fig = build_confusion_matrix_figure(y_test, y_pred, labels)
    st.pyplot(fig)
    plt.close(fig)

def render_feature_importance(
    model: RandomForestClassifier, vectorizer: TfidfVectorizer, top_n: int = TOP_N_FEATURES
) -> None:
    '''
    Render a bar chart of the top N most important TF-IDF features.
    
    Args:
        model: Trained Random Forest classifier.
        vectorizer: Fitted TF-IDF vectorizer.
        top_n: Number of top features to display.
    '''
    st.subheader('Feature Importance')
    feature_names = np.array(vectorizer.get_feature_names_out())
    importances = model.feature_importances_
    fig = build_feature_importance_figure(feature_names, importances, top_n)
    st.pyplot(fig)
    plt.close(fig)

def render_dashboard_tab(
    model: RandomForestClassifier, vectorizer: TfidfVectorizer, y_test, X_test_vec, metadata: dict,
) -> None:
    '''
    Render the dashboard tab containing all three visualizations.
    
    Args:
        model: Trained Random Forest classifier.
        vectorizer: Fitted TF-IDF vectorizer.
        y_test: True labels for the test set.
        X_test_vec: TF-IDF feature matrix for the test set.
    '''
    st.header('Model Dashboard')
    render_model_info(metadata)
    render_recent_predictions()
    st.divider()
    render_distribution_chart(y_test)
    render_confusion_matrix(model, X_test_vec, y_test)
    render_feature_importance(model, vectorizer)