from __future__ import annotations

import streamlit as st


def check_password() -> bool:
    '''
    Render a password gate and return whether access should be granted.
    
    Returns:
        True if the correct password has been entered, otherwise False.
    '''
    if st.session_state.get('authenticated', False):
        return True
    
    st.title('IT Ticket Classifier - Login')
    password_input = st.text_input('Password', type='password')

    if st.button('Log In'):
        if password_input == st.secrets['password']:
            st.session_state['authenticated'] = True
            st.rerun()
        else:
            st.error('Incorrect password')

    return False