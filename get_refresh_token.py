from google_auth_oauthlib.flow import InstalledAppFlow
import os
import json

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/adwords']

def main():
    # Print current directory to help with troubleshooting
    print("Current directory:", os.getcwd())
    print("Looking for credentials file...")
    
    # List files in current directory
    files = [f for f in os.listdir() if f.startswith('client_secret') and f.endswith('.json')]
    
    if not files:
        print("No credentials file found! Please ensure the client_secret*.json file is in this directory.")
        return
        
    credentials_file = files[0]
    print(f"Found credentials file: {credentials_file}")

    # Load client secrets from the downloaded JSON file
    flow = InstalledAppFlow.from_client_secrets_file(
        credentials_file,
        scopes=SCOPES
    )

    # Run the OAuth flow
    credentials = flow.run_local_server(port=8501)

    # Print the refresh token
    print("\nYour refresh token is:", credentials.refresh_token)
    print("\nPlease save this token securely and use it in your Streamlit secrets.")

if __name__ == '__main__':
    main()