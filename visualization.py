import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
from wordcloud import WordCloud
import logging

logging.basicConfig(level=logging.INFO)

def create_word_cloud(df_ngrams: pd.DataFrame):
    """
    Create and display a word cloud visualization of n-grams.
    :param df_ngrams: DataFrame containing 'ngram' and 'frequency' columns
    """
    try:
        # Create word frequency dictionary
        word_freq = dict(zip(df_ngrams['ngram'], df_ngrams['frequency']))

        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            prefer_horizontal=0.7
        ).generate_from_frequencies(word_freq)

        # Display using matplotlib
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")
        logging.error(f"Word cloud generation failed: {str(e)}")

def plot_dual_axis_chart(df_metrics: pd.DataFrame):
    """Option 1: Dual Axis Bar/Line Chart"""
    try:
        base = alt.Chart(df_metrics).encode(
            y=alt.Y('N-gram:N', sort='-x')
        )

        # Bar chart for frequency
        bars = base.mark_bar(color='#74b9ff').encode(
            x=alt.X('Frequency:Q', title='Frequency'),
            tooltip=['N-gram', 'Frequency', 'CPA']
        )

        # Line chart for CPA
        line = base.mark_line(color='#ff7675', point=True).encode(
            x=alt.X('CPA_float:Q', title='CPA ($)'),
            tooltip=['N-gram', 'CPA']
        )

        # Combine charts
        chart = alt.layer(bars, line).resolve_scale(
            x='independent'
        ).properties(
            title='Frequency and CPA by N-gram',
            height=400
        )

        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating dual axis chart: {str(e)}")
        logging.error(f"Dual axis chart creation failed: {str(e)}")

def plot_color_coded_bars(df_metrics: pd.DataFrame):
    """Option 2: Color-Coded Bars based on CPA"""
    try:
        def get_color(cpa):
            if cpa <= 50: return '#00b894'  # Green
            if cpa <= 100: return '#fdcb6e'  # Yellow
            return '#ff7675'  # Red

        df_metrics['color'] = df_metrics['CPA_float'].apply(get_color)

        chart = alt.Chart(df_metrics).mark_bar().encode(
            y=alt.Y('N-gram:N', sort='-x'),
            x='Frequency:Q',
            color=alt.Color('color:N', scale=None),
            tooltip=['N-gram', 'Frequency', 'CPA']
        ).properties(
            title='N-grams by Frequency (Color = CPA Range)',
            height=400
        )

        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating color-coded bars: {str(e)}")
        logging.error(f"Color-coded bars creation failed: {str(e)}")

def plot_side_by_side_bars(df_metrics: pd.DataFrame):
    """Option 3: Side-by-Side Bars"""
    try:
        # Melt the dataframe to get Frequency and CPA side by side
        df_melted = pd.melt(
            df_metrics,
            id_vars=['N-gram'],
            value_vars=['Frequency', 'CPA_float'],
            var_name='Metric',
            value_name='Value'
        )

        chart = alt.Chart(df_melted).mark_bar().encode(
            y=alt.Y('N-gram:N', sort='-x'),
            x='Value:Q',
            color='Metric:N',
            row=alt.Row('Metric:N', title=None),
            tooltip=['N-gram', 'Value']
        ).properties(
            title='Frequency and CPA Comparison',
            height=200
        )

        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating side by side bars: {str(e)}")
        logging.error(f"Side by side bars creation failed: {str(e)}")

def plot_heatmap(df_metrics: pd.DataFrame):
    """Option 4: Heat Map Grid"""
    try:
        # Melt the dataframe for heatmap format
        df_melted = pd.melt(
            df_metrics,
            id_vars=['N-gram'],
            value_vars=['Frequency', 'CPA_float'],
            var_name='Metric',
            value_name='Value'
        )

        chart = alt.Chart(df_melted).mark_rect().encode(
            y=alt.Y('N-gram:N', sort='-x'),
            x='Metric:N',
            color='Value:Q',
            tooltip=['N-gram', 'Metric', 'Value']
        ).properties(
            title='Heat Map of Metrics',
            height=400
        )

        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating heatmap: {str(e)}")
        logging.error(f"Heatmap creation failed: {str(e)}")

def plot_bubble_chart(df_metrics: pd.DataFrame):
    """Option 5: Bubble Chart"""
    try:
        chart = alt.Chart(df_metrics).mark_circle().encode(
            x='Frequency:Q',
            y='CPA_float:Q',
            size='Total_Cost_float:Q',
            color=alt.Color('N-gram:N', legend=None),
            tooltip=['N-gram', 'Frequency', 'CPA', 'Total Cost']
        ).properties(
            title='Bubble Chart (Size = Total Cost)',
            width=600,
            height=400
        )

        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating bubble chart: {str(e)}")
        logging.error(f"Bubble chart creation failed: {str(e)}")

@st.cache_data
def plot_ngram_analysis(df_ngrams: pd.DataFrame, df_original: pd.DataFrame, search_term_col: str, cost_col: str, conversions_col: str, top_k: int = 20):
    """
    Display multiple visualizations of n-gram analysis
    """
    try:
        # Get top k n-grams
        df_top = df_ngrams.head(top_k).copy()

        # Calculate metrics for each n-gram
        metrics = []
        for ngram in df_top['ngram']:
            mask = df_original[search_term_col].astype(str).str.contains(str(ngram), case=False, na=False)
            total_cost = df_original.loc[mask, cost_col].apply(lambda x: float(str(x).replace('$', '').replace(',', ''))).sum()
            total_conversions = df_original.loc[mask, conversions_col].astype(float).sum()
            cpa = total_cost / total_conversions if total_conversions > 0 else 0

            metrics.append({
                'N-gram': str(ngram),
                'Frequency': int(df_top[df_top['ngram'] == ngram]['frequency'].iloc[0]),
                'Total Cost': float(total_cost),
                'Total Conversions': int(total_conversions),
                'CPA': float(cpa)
            })

        # Create metrics DataFrame
        df_metrics = pd.DataFrame(metrics)

        # Add float columns for visualizations
        df_metrics['CPA_float'] = df_metrics['CPA'].astype(float)
        df_metrics['Total_Cost_float'] = df_metrics['Total Cost'].astype(float)

        # Format currency columns for display
        df_metrics['Total Cost'] = df_metrics['Total Cost'].apply(lambda x: f"${x:,.2f}")
        df_metrics['CPA'] = df_metrics['CPA'].apply(lambda x: f"${x:,.2f}" if x > 0 else "$0.00")

        # Display metrics table
        st.dataframe(
            df_metrics[['N-gram', 'Frequency', 'Total Cost', 'Total Conversions', 'CPA']],
            column_config={
                'N-gram': st.column_config.TextColumn("N-gram"),
                'Frequency': st.column_config.NumberColumn("Frequency"),
                'Total Cost': st.column_config.TextColumn("Total Cost"),
                'Total Conversions': st.column_config.NumberColumn("Total Conversions"),
                'CPA': st.column_config.TextColumn("CPA")
            },
            hide_index=True
        )

        # Display all visualization options
        st.subheader("1. Dual Axis Bar/Line Chart")
        plot_dual_axis_chart(df_metrics)

        st.subheader("2. Color-Coded Bars (by CPA)")
        plot_color_coded_bars(df_metrics)

        st.subheader("3. Side-by-Side Bars")
        plot_side_by_side_bars(df_metrics)

        st.subheader("4. Heat Map")
        plot_heatmap(df_metrics)

        st.subheader("5. Bubble Chart")
        plot_bubble_chart(df_metrics)

    except Exception as e:
        st.error(f"Error during visualization processing: {str(e)}")
        logging.error(f"Visualization processing failed: {str(e)}")
