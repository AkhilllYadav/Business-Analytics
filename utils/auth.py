import streamlit as st
import hmac
import time

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["username"], "user@123") and \
           hmac.compare_digest(st.session_state["password"], "password123"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password
        else:
            st.session_state["password_correct"] = False

    # First create a place to store the password check
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    # Show inputs for username and password if not logged in
    if not st.session_state["password_correct"]:
        st.markdown("""
            <div style='text-align: center; padding: 1rem;'>
                <h1>Login Required</h1>
                <p style='color: var(--text-secondary);'>Please enter your credentials to access the dashboard</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.button("Login", on_click=password_entered)
        
        return False
    
    return True
