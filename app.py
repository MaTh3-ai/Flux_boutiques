# app.py
import streamlit as st

# Configuration Streamlit
st.set_page_config(layout="wide", page_title="Prévisions de fréquentation")

# Initialisation de la session state
if 'page' not in st.session_state:
    st.session_state.page = 'selector'

# Navigation principale
def main():
    if st.session_state.page == 'selector':
        from app.pages.selector import selector_page
        selector_page()
    elif st.session_state.page == 'prediction':
        from app.pages.predictions import predictions_page
        predictions_page()
    elif st.session_state.page == 'update_model':
        from app.pages.update_model import update_model_page
        update_model_page()
    elif st.session_state.page == 'update_all_models':
        from app.pages.update_all_models import update_all_models_page
        update_all_models_page()
    elif st.session_state.page == 'manage_boutiques':
        from app.pages.manage_boutiques import manage_boutiques_page
        manage_boutiques_page()
    elif st.session_state.page == 'test':
        from app.pages.test import test
        test()

    else:
        st.error("Page inconnue.")

if __name__ == "__main__":
    main()
