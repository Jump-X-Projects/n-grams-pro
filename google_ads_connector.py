from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
import pandas as pd
import streamlit as st
import re
import logging

logger = logging.getLogger(__name__)

class GoogleAdsConnector:
    def __init__(self):
        self.client = None

    def validate_customer_id(self, customer_id):
        """Validate customer ID format"""
        logger.info(f"Validating customer ID: {customer_id}")

        if not customer_id:
            return False, "Customer ID cannot be empty"

        # Remove any dashes or spaces and ensure it's a string
        customer_id = str(customer_id).strip()
        customer_id = re.sub(r'[-\s]', '', customer_id)

        if not customer_id.isdigit():
            return False, "Customer ID must contain only numbers"

        # Don't pad - ensure it's exactly 10 digits
        if len(customer_id) != 10:
            return False, f"Customer ID must be exactly 10 digits (got {len(customer_id)} digits)"

        return True, customer_id

    def initialize_client(self):
        """Initialize Google Ads client with credentials from Streamlit secrets"""
        try:
            logger.info("Initializing Google Ads client...")

            # Get credentials from environment variables
            developer_token = st.secrets["google_ads"]["developer_token"]
            client_id = st.secrets["google_ads"]["client_id"]
            client_secret = st.secrets["google_ads"]["client_secret"]
            refresh_token = st.secrets["google_ads"]["refresh_token"]
            login_customer_id = st.secrets["google_ads"]["login_customer_id"]

            # Ensure login_customer_id is 10 digits
            if len(login_customer_id) < 10:
                login_customer_id = login_customer_id.zfill(10)

            logger.info(f"Using login_customer_id: {login_customer_id}")

            credentials = {
                'developer_token': developer_token,
                'client_id': client_id,
                'client_secret': client_secret,
                'refresh_token': refresh_token,
                'login_customer_id': login_customer_id,
                'use_proto_plus': True
            }

            logger.info("Creating Google Ads client...")
            self.client = GoogleAdsClient.load_from_dict(credentials)
            logger.info("Google Ads client created successfully")
            return True

        except KeyError as e:
            st.error(f"Missing required Google Ads credential: {str(e)}")
            logger.error(f"Missing credential: {str(e)}")
            return False
        except Exception as e:
            st.error(f"Failed to initialize Google Ads client: {str(e)}")
            logger.error(f"Client initialization error: {str(e)}", exc_info=True)
            return False

    def get_search_terms_report(self, customer_id, date_range='LAST_30_DAYS'):
        """Fetch search terms report from Google Ads"""
        logger.info(f"Getting search terms report for customer ID: {customer_id}")

        # Validate customer_id first
        is_valid, result = self.validate_customer_id(customer_id)
        if not is_valid:
            st.error(f"Invalid customer ID: {result}")
            logger.error(f"Customer ID validation failed: {result}")
            return None

        customer_id = result  # Use validated customer ID

        if not self.client:
            logger.info("No client found, attempting to initialize...")
            if not self.initialize_client():
                return None

        try:
            logger.info("Getting Google Ads service...")
            ga_service = self.client.get_service("GoogleAdsService")

            query = """
                SELECT
                    search_term_view.search_term,
                    metrics.cost_micros,
                    metrics.conversions,
                    metrics.impressions,
                    metrics.clicks
                FROM search_term_view
                WHERE segments.date DURING {date_range}
            """.format(date_range=date_range)

            logger.info(f"Executing query for customer ID {customer_id}")
            search_terms_data = []
            stream = ga_service.search_stream(
                customer_id=customer_id,
                query=query
            )

            for batch in stream:
                for row in batch.results:
                    search_terms_data.append({
                        'search_term': row.search_term_view.search_term,
                        'cost': row.metrics.cost_micros / 1000000,
                        'conversions': row.metrics.conversions,
                        'impressions': row.metrics.impressions,
                        'clicks': row.metrics.clicks
                    })

            if not search_terms_data:
                st.warning("No search terms data found for the specified date range")
                logger.warning("No data found in search terms report")
                return None

            logger.info(f"Successfully retrieved {len(search_terms_data)} search terms")
            return pd.DataFrame(search_terms_data)

        except GoogleAdsException as ex:
            for error in ex.failure.errors:
                error_message = f'Google Ads API error: {error.message}'
                st.error(error_message)
                logger.error(error_message)
            return None
        except Exception as e:
            error_message = f"Error fetching search terms: {str(e)}"
            st.error(error_message)
            logger.error(error_message, exc_info=True)
            return None
