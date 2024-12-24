import streamlit as st
import logging
from logging.handlers import RotatingFileHandler

# Configure logging with size limits and rotation
logging.basicConfig(
    level=logging.INFO,  # Change to INFO level for production
    handlers=[
        RotatingFileHandler(
            'app.log',
            maxBytes=1024*1024,  # 1MB file size
            backupCount=5
        )
    ]
)
logger = logging.getLogger(__name__)

def check_credentials():
    """Check if all required credentials are present"""
    try:
        required_secrets = [
            "google_ads.developer_token",
            "google_ads.client_id",
            "google_ads.client_secret",
            "google_ads.refresh_token",
            "google_ads.login_customer_id"
        ]
        
        for secret in required_secrets:
            if not st.secrets.get(secret.split('.')[0], {}).get(secret.split('.')[1]):
                raise KeyError(f"Missing required secret: {secret}")
        return True
    except Exception as e:
        logger.error(f"Credentials check failed: {str(e)}")
        return False

def main():
    try:
        st.set_page_config(
            page_title="N-Gram Analysis Tool (Pro)",
            page_icon="📊",
            initial_sidebar_state="expanded"
        )

        st.title("N-Gram Analysis Tool (Pro Version)")
        st.write("Analyze search terms and performance metrics from CSV or directly from Google Ads.")

        # Data Source Selection
        data_source = st.radio(
            "Choose Data Source",
            ["Upload CSV", "Google Ads Data"],
            help="Select where to get your data from"
        )

        if data_source == "Google Ads Data":
            # Check credentials before showing Google Ads options
            if not check_credentials():
                st.error("Google Ads credentials are not properly configured. Please contact support.")
                return

            with st.expander("Google Ads Configuration", expanded=True):
                customer_id = st.text_input(
                    "Customer ID (without dashes)",
                    help="Your Google Ads account ID"
                )

                date_range = st.selectbox(
                    "Date Range",
                    ["LAST_30_DAYS", "LAST_7_DAYS", "LAST_14_DAYS", "LAST_90_DAYS"],
                    help="Select the time period for search terms"
                )

                if st.button("Fetch Google Ads Data"):
                    with st.spinner("Connecting to Google Ads..."):
                        from google_ads_connector import GoogleAdsConnector
                        connector = GoogleAdsConnector()
                        
                        if connector.initialize_client():
                            st.success("Connected to Google Ads!")
                            if customer_id:
                                data = connector.get_search_terms_report(customer_id, date_range)
                                if data is not None:
                                    st.write("Data fetched successfully!")
                                    st.dataframe(data)
                        else:
                            st.error("Failed to connect to Google Ads. Please check your credentials.")
        else:
            st.info("CSV upload functionality coming soon!")
            uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
            if uploaded_file is not None:
                st.warning("CSV processing not implemented yet.")

    except Exception as e:
        logger.error(f"Error in main app: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred. Please try again later or contact support.")

if __name__ == "__main__":
    main()