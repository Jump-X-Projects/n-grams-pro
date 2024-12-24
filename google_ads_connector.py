from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
import pandas as pd
import streamlit as st
import re
import logging
import sys
from functools import wraps
import time

logger = logging.getLogger(__name__)

def timeout_handler(timeout_duration=30):
    """Decorator to handle timeouts"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = None
            
            while time.time() - start_time < timeout_duration:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    if time.time() - start_time >= timeout_duration:
                        raise TimeoutError(f"Operation timed out after {timeout_duration} seconds")
                    logger.warning(f"Retrying operation: {str(e)}")
                    time.sleep(1)
            
            raise TimeoutError(f"Operation timed out after {timeout_duration} seconds")
        return wrapper
    return decorator

class GoogleAdsConnector:
    def __init__(self):
        self.client = None
        self.api_version = 'v14'

    def validate_customer_id(self, customer_id):
        """Validate customer ID format"""
        logger.info(f"Validating customer ID: {customer_id}")

        if not customer_id:
            return False, "Customer ID cannot be empty"

        customer_id = str(customer_id).strip()
        customer_id = re.sub(r'[-\s]', '', customer_id)

        if not customer_id.isdigit():
            return False, "Customer ID must contain only numbers"

        if len(customer_id) != 10:
            return False, f"Customer ID must be exactly 10 digits long (got {len(customer_id)} digits)"

        return True, customer_id

    @timeout_handler(timeout_duration=30)
    def initialize_client(self):
        """Initialize Google Ads client with credentials from Streamlit secrets"""
        try:
            # Verify all required secrets are present
            required_secrets = [
                "developer_token",
                "client_id",
                "client_secret",
                "refresh_token",
                "login_customer_id"
            ]
            
            credentials = {}
            for secret in required_secrets:
                value = st.secrets["google_ads"].get(secret)
                if not value:
                    raise KeyError(f"Missing required Google Ads credential: {secret}")
                credentials[secret] = value

            credentials['use_proto_plus'] = True
            credentials['version'] = self.api_version

            logger.info("Initializing Google Ads client...")
            self.client = GoogleAdsClient.load_from_dict(credentials)
            logger.info("Successfully created Google Ads client")
            return True

        except KeyError as e:
            st.error(f"Configuration error: {str(e)}")
            logger.error("Missing required credentials", exc_info=True)
            return False
        except Exception as e:
            st.error(f"Failed to initialize Google Ads client: {str(e)}")
            logger.error("Client initialization error", exc_info=True)
            return False

    @timeout_handler(timeout_duration=60)
    def get_search_terms_report(self, customer_id, date_range='LAST_30_DAYS'):
        """Fetch search terms report from Google Ads"""
        try:
            # Validate customer_id
            is_valid, result = self.validate_customer_id(customer_id)
            if not is_valid:
                st.error(f"Invalid customer ID: {result}")
                return None

            customer_id = result
            
            if not self.client:
                logger.info("No client found, attempting to initialize...")
                if not self.initialize_client():
                    return None

            ga_service = self.client.get_service("GoogleAdsService", version=self.api_version)

            # Test query with timeout
            test_query = """
                SELECT
                    campaign.id,
                    campaign.name
                FROM campaign
                LIMIT 1
            """

            try:
                response = ga_service.search(
                    customer_id=customer_id,
                    query=test_query
                )
            except Exception as e:
                logger.error(f"Test query failed: {str(e)}")
                st.error("Failed to connect to your Google Ads account. Please verify your customer ID.")
                return None

            # Main query for search terms
            query = f"""
                SELECT
                    search_term_view.search_term,
                    metrics.cost_micros,
                    metrics.conversions,
                    metrics.impressions,
                    metrics.clicks
                FROM search_term_view
                WHERE segments.date DURING {date_range}
            """

            search_terms_data = []
            try:
                response = ga_service.search(
                    customer_id=customer_id,
                    query=query
                )

                for row in response:
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

                return pd.DataFrame(search_terms_data)

            except GoogleAdsException as ex:
                for error in ex.failure.errors:
                    error_msg = f"Google Ads API error: {error.message}"
                    if hasattr(error, 'details'):
                        error_msg += f"\nDetails: {error.details}"
                    st.error(error_msg)
                    logger.error(error_msg)
                return None

        except TimeoutError as te:
            st.error("The request took too long to complete. Please try again.")
            logger.error(f"Timeout error: {str(te)}")
            return None
        except Exception as e:
            st.error(f"Error fetching search terms: {str(e)}")
            logger.error("Error in get_search_terms_report", exc_info=True)
            return None