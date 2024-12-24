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

class GoogleAdsConnector:
    def __init__(self):
        self.client = None
        self.api_version = 'v14'
        self.mcc_id = None

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

    def initialize_client(self):
        """Initialize Google Ads client with credentials from Streamlit secrets"""
        try:
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

            # Store MCC ID
            self.mcc_id = str(credentials["login_customer_id"]).strip()
            self.mcc_id = re.sub(r'[-\s]', '', self.mcc_id)

            credentials['use_proto_plus'] = True
            credentials['version'] = self.api_version

            logger.info(f"Initializing Google Ads client for MCC: {self.mcc_id}")
            self.client = GoogleAdsClient.load_from_dict(credentials)
            logger.info("Successfully created Google Ads client")
            return True

        except Exception as e:
            st.error(f"Failed to initialize Google Ads client: {str(e)}")
            logger.error("Client initialization error", exc_info=True)
            return False

    def get_accessible_accounts(self):
        """Get list of all accessible accounts under the MCC"""
        try:
            if not self.client:
                logger.error("Client not initialized")
                return []

            ga_service = self.client.get_service("GoogleAdsService", version=self.api_version)
            
            # Simpler query to just get basic account information
            query = """
                SELECT
                    customer_client.id,
                    customer_client.descriptive_name
                FROM customer_client
            """
            
            logger.info(f"Fetching accounts for MCC: {self.mcc_id}")
            
            try:
                # Execute query from MCC account
                response = ga_service.search(
                    customer_id=self.mcc_id,
                    query=query
                )
                
                accounts = []
                for row in response:
                    account = {
                        'id': str(row.customer_client.id),
                        'name': row.customer_client.descriptive_name or f"Account {row.customer_client.id}"
                    }
                    logger.info(f"Found account: {account}")
                    accounts.append(account)
                
                logger.info(f"Total accounts found: {len(accounts)}")
                return accounts
                
            except GoogleAdsException as ex:
                logger.error("Google Ads API Exception:")
                for error in ex.failure.errors:
                    logger.error(f"\tError with Message: {error.message}")
                    if hasattr(error, 'details'):
                        logger.error(f"\tDetails: {error.details}")
                st.error(f"Error accessing Google Ads accounts: {ex.failure.errors[0].message}")
                return []
                
        except Exception as e:
            logger.error(f"Unexpected error getting accessible accounts: {str(e)}")
            st.error("Failed to fetch accounts. Please check the application logs.")
            return []

    def get_search_terms_report(self, customer_id, date_range='LAST_30_DAYS'):
        """Fetch search terms report from Google Ads"""
        try:
            # Validate customer_id
            is_valid, result = self.validate_customer_id(customer_id)
            if not is_valid:
                st.error(f"Invalid customer ID: {result}")
                return None

            customer_id = result
            logger.info(f"Fetching search terms for account: {customer_id}")

            ga_service = self.client.get_service("GoogleAdsService", version=self.api_version)

            # Query for search terms
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

            try:
                response = ga_service.search(
                    customer_id=customer_id,
                    query=query
                )

                search_terms_data = []
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
                    st.error(error_msg)
                    logger.error(error_msg)
                return None

        except Exception as e:
            st.error(f"Error fetching search terms: {str(e)}")
            logger.error("Error in get_search_terms_report", exc_info=True)
            return None