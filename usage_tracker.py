import streamlit as st
from datetime import datetime
import logging

# Configure logging to print to console (visible in Streamlit Cloud logs)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def track_login(email, role, name="Unknown"):
    """
    Logs a user login event.
    
    1. Prints to console (Standard Logging) - Viewable in Streamlit Cloud Dashboard logs.
    2. (Optional) Appends to Google Sheets if configured in secrets.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Console Logging
    log_message = f"LOGIN_EVENT | Time: {timestamp} | User: {email} | Role: {role} | Name: {name}"
    logger.info(log_message)
    print(log_message) # Ensure it goes to stdout as well

    # 2. Google Sheets Logging (Persistent Storage)
    # Requires [google_sheets] section in .streamlit/secrets.toml
    if "google_sheets" in st.secrets:
        try:
            _log_to_google_sheet(timestamp, email, role, name)
        except Exception as e:
            logger.error(f"Failed to log to Google Sheets: {e}")

def _log_to_google_sheet(timestamp, email, role, name):
    """
    Internal function to append row to Google Sheet.
    Uses google-api-python-client (already in requirements).
    """
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    # Load credentials from secrets
    # Expected format in secrets.toml:
    # [google_sheets]
    # spreadsheet_id = "..."
    # type = "service_account"
    # project_id = "..."
    # ... (other service account fields)
    
    try:
        # Construct credentials dictionary from secrets
        # We assume the secrets structure mirrors the service account JSON
        creds_dict = dict(st.secrets["google_sheets"])
        
        # Remove spreadsheet_id from creds_dict as it's not part of auth
        spreadsheet_id = creds_dict.pop("spreadsheet_id", None)
        
        if not spreadsheet_id:
            logger.warning("No 'spreadsheet_id' found in secrets[google_sheets].")
            return

        scopes = ['https://www.googleapis.com/auth/spreadsheets']
        creds = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=scopes)

        service = build('sheets', 'v4', credentials=creds)
        sheet = service.spreadsheets()

        # Data to append
        values = [[timestamp, email, role, name]]
        body = {'values': values}

        # Append to 'Sheet1' (or first sheet)
        range_name = 'Sheet1!A:D' 
        
        result = sheet.values().append(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption='USER_ENTERED',
            body=body
        ).execute()
        
        logger.info(f"Successfully logged to Google Sheet: {result.get('updates', {}).get('updatedCells')} cells updated.")

    except Exception as e:
        logger.error(f"Google Sheets API Error: {e}")
        # Do not stop the app flow if logging fails
