from google_auth_oauthlib.flow import InstalledAppFlow
import json

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/adwords']

def main():
    # Load client secrets from the downloaded JSON file
    flow = InstalledAppFlow.from_client_secrets_file(
        'client_secret_506501675497-io98mnuloij9cv5uo9s27mvggqm82sl1.apps.googleusercontent.com (1).json',
        scopes=SCOPES
    )

    # Run the OAuth flow
    credentials = flow.run_local_server(port=8501)

    # Print the refresh token
    print("\nYour refresh token is:", credentials.refresh_token)
    print("\nPlease save this token securely and use it in your Streamlit secrets.")

if __name__ == '__main__':
    main()