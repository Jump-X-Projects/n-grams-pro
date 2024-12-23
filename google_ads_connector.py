from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
import pandas as pd
import streamlit as st

class GoogleAdsConnector:
    def __init__(self):
        self.client = None

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
            self.client = GoogleAdsClient.load_from_dict(credentials)
            return True
        except Exception as e:
            st.error(f"Failed to initialize Google Ads client: {str(e)}")
            return False

    def get_search_terms_report(self, customer_id, date_range='LAST_30_DAYS'):
        """Fetch search terms report from Google Ads"""
        if not self.client:
            if not self.initialize_client():
                return None

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

        try:
            search_terms_data = []
            stream = ga_service.search_stream(
                customer_id=customer_id,
                query=query
            )

            for batch in stream:
                for row in batch.results:
                    search_terms_data.append({
                        'search_term': row.search_term_view.search_term,
                        'cost': row.metrics.cost_micros / 1000000,  # Convert to actual currency
                        'conversions': row.metrics.conversions,
                        'impressions': row.metrics.impressions,
                        'clicks': row.metrics.clicks
                    })

            return pd.DataFrame(search_terms_data)

        except GoogleAdsException as ex:
            for error in ex.failure.errors:
                st.error(f'Google Ads API error: {error.message}')
            return None
        except Exception as e:
            st.error(f"Error fetching search terms: {str(e)}")
            return None
