"""
User Role Management System
Handles premium and basic user tier assignments
"""

import os
import streamlit as st


def load_premium_users(file_path='premium_users.txt'):
    """
    Load list of premium user emails from configuration.
    
    Priority:
    1. st.secrets['premium']['users'] (recommended for production)
    2. premium_users.txt file (fallback for local dev)
    
    Returns:
        set: Set of premium user email addresses (lowercase)
    """
    premium_emails = set()
    
    # First, try loading from secrets.toml (recommended)
    try:
        if "premium" in st.secrets and "users" in st.secrets["premium"]:
            users_list = st.secrets["premium"]["users"]
            # Handle both list and single string
            if isinstance(users_list, list):
                for email in users_list:
                    if email and isinstance(email, str):
                        premium_emails.add(email.strip().lower())
            elif isinstance(users_list, str):
                premium_emails.add(users_list.strip().lower())
            
            if premium_emails:
                return premium_emails
    except Exception:
        # If secrets not available, fall through to file method
        pass
    
    # Fallback: Load from txt file (backwards compatibility)
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    # Remove whitespace and skip comments/empty lines
                    email = line.strip()
                    if email and not email.startswith('#'):
                        premium_emails.add(email.lower())
        except Exception as e:
            st.warning(f"Could not load premium users from file: {e}")
    
    return premium_emails


def get_user_role(email):
    """
    Determine user role based on email address.
    
    Args:
        email (str): User's email address
    
    Returns:
        str: 'premium' or 'basic'
    """
    if not email:
        return 'basic'
    
    premium_users = load_premium_users()
    
    if email.lower() in premium_users:
        return 'premium'
    else:
        return 'basic'


def is_premium_user():
    """
    Check if currently logged-in user has premium access.
    
    Returns:
        bool: True if user is premium, False otherwise
    """
    # Check if user is authenticated
    if not st.session_state.get("authentication_status"):
        return False
    
    # Get user role from session state (set during login)
    user_role = st.session_state.get("user_role", "basic")
    return user_role == "premium"


def require_premium_feature(feature_name="This feature"):
    """
    Display a message indicating a feature requires premium access.
    
    Args:
        feature_name (str): Name of the premium feature
    """
    st.info(f"🔒 **{feature_name}** is available for premium users only.")
    st.caption("Contact your administrator to upgrade your account.")


def show_role_badge():
    """Display user role badge in the UI."""
    if st.session_state.get("authentication_status"):
        role = st.session_state.get("user_role", "basic")
        
        if role == "premium":
            st.sidebar.markdown("🌟 **Premium User**")
        else:
            st.sidebar.markdown("👤 **Basic User**")
