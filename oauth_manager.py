import os
import google_auth_oauthlib.flow
from googleapiclient.discovery import build
import streamlit as st

class GoogleAuthManager:
    def __init__(self, client_secrets_file='client_secret.json', scopes=None):
        if scopes is None:
            self.scopes = [
                "openid",
                "https://www.googleapis.com/auth/userinfo.profile",
                "https://www.googleapis.com/auth/userinfo.email"
            ]
        else:
            self.scopes = scopes
        
        self.client_secrets_file = client_secrets_file

    def get_flow(self):
        """Creates and returns the OAuth flow object."""
        # Check st.secrets first (better for deployment and local dev)
        if "google" in st.secrets:
            try:
                client_config = {
                    "web": {
                        "client_id": st.secrets["google"]["client_id"],
                        "client_secret": st.secrets["google"]["client_secret"],
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": [st.secrets["google"]["redirect_uri"]],
                    }
                }
                flow = google_auth_oauthlib.flow.Flow.from_client_config(
                    client_config,
                    scopes=self.scopes
                )
            except KeyError as e:
                st.error(f"Missing Google OAuth configuration in secrets.toml: {e}")
                st.stop()
        elif os.path.exists(self.client_secrets_file):
            flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
                self.client_secrets_file,
                scopes=self.scopes
            )
        else:
            st.error("Google OAuth credentials not found. Please provide 'client_secret.json' or configure secrets.toml with [google] section.")
            st.stop()

        # Set redirect URI from secrets or use default
        if "google" in st.secrets and "redirect_uri" in st.secrets["google"]:
             flow.redirect_uri = st.secrets["google"]["redirect_uri"]
        else:
             flow.redirect_uri = "http://localhost:8501" 
        return flow

    def get_login_url(self):
        """Generates the authorization URL."""
        flow = self.get_flow()
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true'
        )
        return authorization_url, state

    def get_token_from_code(self, code):
        """Exchanges the authorization code for tokens."""
        flow = self.get_flow()
        flow.fetch_token(code=code)
        credentials = flow.credentials
        return credentials

    def get_user_info(self, credentials):
        """Fetches user information using the credentials."""
        service = build('oauth2', 'v2', credentials=credentials)
        user_info = service.userinfo().get().execute()
        return user_info
