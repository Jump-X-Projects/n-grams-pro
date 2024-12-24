from google_auth_oauthlib.flow import InstalledAppFlow
import os
import json

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/adwords']

def main():
    print("Current directory:", os.getcwd())
    print("Looking for credentials file...")

    files = [f for f in os.listdir() if f.startswith('client_secret') and f.endswith('.json')]

    if not files:
        print("No credentials file found! Please ensure the client_secret*.json file is in this directory.")
        return

    credentials_file = files[0]
    print(f"Found credentials file: {credentials_file}")

    # Create flow using the client secrets file
    flow = InstalledAppFlow.from_client_secrets_file(
        credentials_file,
        scopes=SCOPES,
        redirect_uri='http://localhost:8080'  # Changed from 8501
    )

    # Run the OAuth flow
    flow.run_local_server(
        port=8080,  # Match the port in redirect_uri
        prompt='consent',
        access_type='offline'
    )

    # Get credentials and print refresh token
    credentials = flow.credentials
    print("\nYour refresh token is:", credentials.refresh_token)
    print("\nPlease save this token securely and use it in your Streamlit secrets.")

if __name__ == '__main__':
    main()
