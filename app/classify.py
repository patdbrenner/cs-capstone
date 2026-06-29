from __future__ import annotations

import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from app.prediction_log import log_prediction
from src.preprocess import clean_text


def render_classification_tab(model: RandomForestClassifier, vectorizer: TfidfVectorizer) -> None:
    '''
    Render the ticket submission form and display the predicted category.
    
    Args:
        model: Trained Random Forest classifier.
        vectorizer: Fitted TF-IDF vectorizer.
    '''
    st.header('Submit a Ticket')

    title = st.text_input('Ticket Title')
    description = st.text_area('Ticket Description')

    if st.button('Classify Ticket'):
        if not title and not description:
            st.warning('Please enter a title or description.')
            return
        
        combined_text = f'{title} {description}'
        cleaned = clean_text(combined_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        proba = model.predict_proba(vectorized)[0]
        classes = model.classes_
        confidence = float(proba.max())

        log_prediction(title, description, prediction, confidence)

        st.success(f'Predicted Category: **{prediction}**')

        top_indices = np.argsort(proba)[-3:][::-1]

        st.subheader('Top 3 Predictions')
        for idx in top_indices:
            st.write(f'{classes[idx]}: {proba[idx]:.2%}')