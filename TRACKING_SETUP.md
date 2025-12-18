# User Usage Tracking Setup

You have two options to track user logins:

## Option 1: Simple Console Logs (Zero Setup)
By default, every login is printed to the application logs.
- **Where to view:** 
  - If running locally: Look at your terminal.
  - If on Streamlit Cloud: Go to your App Dashboard -> Click the "..." menu -> **Manage app** -> **Logs**.
- **What you see:** `LOGIN_EVENT | Time: ... | User: ... | Role: ...`
- **Pros:** Easiest, no configuration.
- **Cons:** Logs are not persistent (they disappear after a while or if the app restarts).

## Option 2: Google Sheets (Persistent History)
To save a permanent history of all logins, connect a Google Sheet.

### Step 1: Create a Google Sheet
1. Create a new Google Sheet (e.g., named "App Usage Logs").
2. Add headers in the first row: `Timestamp`, `Email`, `Role`, `Name`.

### Step 2: Create a Service Account
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (or use your existing one).
3. Enable the **Google Sheets API**.
4. Go to **IAM & Admin** -> **Service Accounts** -> **Create Service Account**.
5. Name it (e.g., "logger").
6. Click **Done**.
7. Click on the newly created service account -> **Keys** -> **Add Key** -> **Create new key** -> **JSON**.
8. A JSON file will download. Open it.

### Step 3: Share the Sheet
1. Open your Google Sheet.
2. Click **Share**.
3. Copy the `client_email` from your JSON file (e.g., `logger@project-id.iam.gserviceaccount.com`).
4. Paste it into the Share dialog and give it **Editor** access.

### Step 4: Configure Secrets
Add the following to your `.streamlit/secrets.toml` file (or the Secrets section in Streamlit Cloud):

```toml
[google_sheets]
spreadsheet_id = "YOUR_SPREADSHEET_ID_FROM_URL"
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "..."
client_email = "..."
client_id = "..."
auth_uri = "..."
token_uri = "..."
auth_provider_x509_cert_url = "..."
client_x509_cert_url = "..."
```
*Note: Copy all the fields from your downloaded JSON file into the `[google_sheets]` section.*
*The `spreadsheet_id` is the long string in your Google Sheet URL: `docs.google.com/spreadsheets/d/THIS_IS_THE_ID/edit`*

> [!IMPORTANT]
> **Private Key Formatting:**
> The `private_key` must include `-----BEGIN PRIVATE KEY-----` and `-----END PRIVATE KEY-----`.
> It also contains `\n` (newline) characters. **Do not remove them.**
> In `secrets.toml`, it should look like this:
> `private_key = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQ...\n-----END PRIVATE KEY-----\n"`
> Or use a multi-line TOML string:
> ```toml
> private_key = """
> -----BEGIN PRIVATE KEY-----
> MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQ...
> ...
> -----END PRIVATE KEY-----
> """
> ```
