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

    # Create the flow using the client secrets file from the Google API Console
    flow = InstalledAppFlow.from_client_secrets_file(
        credentials_file,
        scopes=SCOPES,
        redirect_uri='http://localhost:8501/callback'  # Specify exact callback path
    )

    # Run the OAuth flow with specific parameters
    flow.run_local_server(
        host='localhost',
        port=8501,
        authorization_prompt_message='Please visit this URL to authorize this application: {url}',
        success_message='The authentication flow has completed. You may close this window.',
        open_browser=True
    )

    # Get credentials and print refresh token
    credentials = flow.credentials
    print("\nYour refresh token is:", credentials.refresh_token)
    print("\nPlease save this token securely and use it in your Streamlit secrets.")

if __name__ == '__main__':
    main()