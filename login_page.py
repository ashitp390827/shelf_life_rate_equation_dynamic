import streamlit as st
from oauth_manager import GoogleAuthManager
import os

def show_login_page():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Hide default Streamlit elements first */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .stApp > header {display: none;}
        
        /* Remove default padding */
        .block-container {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            max-width: 100% !important;
        }
        
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            font-family: 'Inter', sans-serif;
        }

        .login-container {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 2rem;
            box-sizing: border-box;
        }

        .login-card {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 24px;
            padding: 3rem 2.5rem;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.5);
            text-align: center;
            max-width: 420px;
            width: 100%;
        }

        .app-title {
            font-size: 2rem;
            font-weight: 700;
            color: #1a202c;
            margin-bottom: 0.5rem;
            letter-spacing: -0.5px;
            line-height: 1.2;
        }

        .app-subtitle {
            font-size: 1rem;
            color: #4a5568;
            margin-bottom: 2rem;
            font-weight: 400;
            line-height: 1.5;
        }

        .google-btn {
            background-color: #ffffff;
            color: #3c4043;
            padding: 14px 24px;
            border-radius: 12px;
            text-decoration: none;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 1rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            transition: all 0.2s ease;
            border: 1px solid #e2e8f0;
            width: 100%;
            box-sizing: border-box;
            cursor: pointer;
        }

        .google-btn:hover {
            background-color: #f8fafc;
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
            transform: translateY(-2px);
            border-color: #cbd5e0;
            text-decoration: none;
        }

        .google-icon-svg {
            margin-right: 12px;
            width: 20px;
            height: 20px;
            flex-shrink: 0;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .login-card {
                padding: 2rem 1.5rem;
            }
            
            .app-title {
                font-size: 1.75rem;
            }
            
            .app-subtitle {
                font-size: 0.95rem;
            }
        }
        
        </style>
        """,
        unsafe_allow_html=True
    )

    # Check for query params (redirect from Google)
    query_params = st.query_params
    code = query_params.get("code")

    if code:
        try:
            auth_manager = GoogleAuthManager()
            credentials = auth_manager.get_token_from_code(code)
            user_info = auth_manager.get_user_info(credentials)
            
            # Get user email
            user_email = user_info.get("email")
            
            # Assign user role based on email
            from user_roles import get_user_role
            user_role = get_user_role(user_email)
            
            # Store user information in session state
            st.session_state["authentication_status"] = True
            st.session_state["name"] = user_info.get("name")
            st.session_state["email"] = user_email
            st.session_state["picture"] = user_info.get("picture")
            st.session_state["user_role"] = user_role
            
            # Track usage
            try:
                from usage_tracker import track_login
                track_login(user_email, user_role, name=user_info.get("name", "Unknown"))
            except Exception as e:
                print(f"Logging failed: {e}")
            
            # Clear the code from the URL to prevent re-execution
            st.query_params.clear()
            st.rerun()
            
        except Exception as e:
            st.error(f"Authentication failed: {e}")
            st.stop()
            
    else:
        auth_manager = GoogleAuthManager()
        try:
            login_url, state = auth_manager.get_login_url()
            
            # Google Icon SVG
            google_svg = '<svg class="google-icon-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/><path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/><path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/><path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/><path fill="none" d="M0 0h48v48H0z"/></svg>'
            
            st.markdown(
                f"""
                <div class="login-container">
                    <div class="login-card">
                        <div class="app-title">Shelf Life Predictor</div>
                        <div class="app-subtitle">Kinetic Modeling & Simulation</div>
                        <a href="{login_url}" class="google-btn" target="_self">
                            {google_svg}
                            Sign in with Google
                        </a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error initializing Google Login: {e}")
            st.info("Please ensure 'client_secret.json' is present in the application directory.")