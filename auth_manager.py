import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import streamlit as st

class AuthManager:
    def __init__(self, config_path='users.yaml'):
        self.config_path = config_path
        self.config = self._load_config()
        
        self.authenticator = stauth.Authenticate(
            self.config['credentials'],
            self.config['cookie']['name'],
            self.config['cookie']['key'],
            self.config['cookie']['expiry_days'],
            # preauthorized=self.config['preauthorized'] 
        )

    def _load_config(self):
        with open(self.config_path) as file:
            return yaml.load(file, Loader=SafeLoader)

    def save_config(self):
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)

    def login_widget(self):
        name, auth_status, username = self.authenticator.login(location='main')
        return name, auth_status, username

    def register_user(self):
        try:
            if self.authenticator.register_user(location='main'):
                self.save_config()
                st.success('User registered successfully')
        except Exception as e:
            st.error(e)

    def logout_widget(self):
        if self.authenticator.logout(location='sidebar'):
            st.rerun()

    def check_authentication(self):
        return st.session_state.get("authentication_status")

    def get_user_name(self):
        return st.session_state.get("name")
