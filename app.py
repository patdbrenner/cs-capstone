from __future__ import annotations

import logging

logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

import streamlit as st

from app.auth import check_password
from app.classify import render_classification_tab
from app.dashboard import render_dashboard_tab
from app.data import load_artifacts, load_metadata, load_test_split

logger = logging.getLogger(__name__)

def main() -> None:
    '''
    Run the streamlit application.
    '''
    st.set_page_config(page_title='IT Ticket Classifier', layout='wide')

    if not check_password():
        return
    
    model, vectorizer = load_artifacts()
    X_test_vec, y_test = load_test_split()
    metadata = load_metadata()

    tab1, tab2 = st.tabs(['Classify Ticket', 'Dashboard'])

    with tab1:
        render_classification_tab(model, vectorizer)

    with tab2:
        render_dashboard_tab(model, vectorizer, y_test, X_test_vec, metadata)

if __name__ == '__main__':
    main()