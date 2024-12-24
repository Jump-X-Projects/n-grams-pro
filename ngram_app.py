import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    try:
        st.title("N-Gram Analysis Tool (Pro Version)")
        st.write("Analyze search terms and performance metrics from CSV or directly from Google Ads.")

        # Data Source Selection
        logger.debug("Setting up data source selection")
        data_source = st.radio(
            "Choose Data Source",
            ["Upload CSV", "Google Ads Data"],
            help="Select where to get your data from"
        )

        if data_source == "Google Ads Data":
            logger.debug("Google Ads Data selected")
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
                    st.info("Testing connection...")
                    from google_ads_connector import GoogleAdsConnector
                    connector = GoogleAdsConnector()
                    logger.debug("Created GoogleAdsConnector instance")

                    if connector.initialize_client():
                        logger.debug("Successfully initialized Google Ads client")
                        st.success("Client initialized successfully!")
                    else:
                        logger.error("Failed to initialize Google Ads client")
        else:
            logger.debug("CSV upload selected")
            st.info("CSV upload functionality will be added here")

    except Exception as e:
        logger.error(f"Error in main app: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
