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

        if len(customer_id) != 10:
            return False, f"Customer ID must be exactly 10 digits long (got {len(customer_id)} digits)"

        return True, customer_id

    def initialize_client(self):
        """Initialize Google Ads client with credentials from Streamlit secrets"""
        try:
            credentials = {
                'developer_token': st.secrets["google_ads"]["developer_token"],
                'client_id': st.secrets["google_ads"]["client_id"],
                'client_secret': st.secrets["google_ads"]["client_secret"],
                'refresh_token': st.secrets["google_ads"]["refresh_token"],
                'login_customer_id': st.secrets["google_ads"]["login_customer_id"],
                'use_proto_plus': True
            }

            logger.info("Credentials loaded from secrets")
            logger.info(f"Using login_customer_id: {credentials['login_customer_id']}")

            self.client = GoogleAdsClient.load_from_dict(credentials)
            logger.info("Successfully created Google Ads client")

            # Test the connection
            service = self.client.get_service("GoogleAdsService")
            logger.info("Successfully got Google Ads service")

            return True

        except Exception as e:
            st.error(f"Failed to initialize Google Ads client: {str(e)}")
            logger.error("Client initialization error", exc_info=True)
            return False

    def get_search_terms_report(self, customer_id, date_range='LAST_30_DAYS'):
        """Fetch search terms report from Google Ads"""
        try:
            # Validate customer_id
            is_valid, result = self.validate_customer_id(customer_id)
            if not is_valid:
                st.error(f"Invalid customer ID: {result}")
                return None

            customer_id = result  # Use validated customer ID
            logger.info(f"Using customer ID: {customer_id}")

            if not self.client:
                logger.info("No client found, attempting to initialize...")
                if not self.initialize_client():
                    return None

            logger.info("Getting Google Ads service...")
            ga_service = self.client.get_service("GoogleAdsService")

            # Start with a simpler test query
            test_query = """
                SELECT
                    customer.id
                FROM customer
                LIMIT 1
            """

            logger.info("Testing connection with simple query...")
            try:
                stream = ga_service.search_stream(
                    customer_id=customer_id,
                    query=test_query
                )
                for batch in stream:
                    logger.info("Successfully executed test query")
            except Exception as e:
                logger.error(f"Test query failed: {str(e)}")
                raise

            # If test succeeds, proceed with main query
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

            logger.info("Executing search terms query...")
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
                return None

            logger.info(f"Successfully retrieved {len(search_terms_data)} search terms")
            return pd.DataFrame(search_terms_data)

        except GoogleAdsException as ex:
            for error in ex.failure.errors:
                error_msg = f"Google Ads API error: {error.message}"
                if hasattr(error, 'details'):
                    error_msg += f"\nDetails: {error.details}"
                st.error(error_msg)
                logger.error(error_msg)
            return None
        except Exception as e:
            st.error(f"Error fetching search terms: {str(e)}")
            logger.error("Error in get_search_terms_report", exc_info=True)
            return None
