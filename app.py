
import streamlit as st

import matplotlib.pyplot as plt
from analysis import sentiment_pipeline
from analysis.sectors import sectors
from analysis import predict
import plotly.express as px
import os
import base64
from analysis import sector_analysis

st.set_page_config(
    page_title="Stock Sentiment Analyzer",
    layout="wide"
)

st.markdown(
    """
    <style>
    .big-font { font-size:24px !important; }
    .metric-label > div { font-size:20px !important; }
    .stCaption { font-size:18px !important; }
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------- Background -------------------------
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

st.markdown(
    """
    <style>
    .stApp {
       background: linear-gradient(135deg, #11998e, #38ef7d);

       background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------- Session State -------------------------
if 'page' not in st.session_state:
    st.session_state.page = 'main'

# ------------------------- Main Page -------------------------
if st.session_state.page == 'main':
    st.title("📈 Stock Sentiment Analyzer")
    st.markdown("""
    <style>
        .reportview-container .main .block-container{
            max-width: 100%;
            padding-left: 2rem;
            padding-right: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

    if st.button("Go to Sector-Wise Sentiment Aggregation"):
        st.session_state.page = 'sector'
        st.rerun()

    st.subheader("🔍 Analyze Individual Stocks")

    sector = st.selectbox("Select Sector", list(sectors.keys()))
    company = st.selectbox("Select Company", list(sectors[sector].keys()))

   
    if st.button("Run Sentiment Analysis"):
        info = sectors[sector][company]
        ticker = info["ticker"]

        st.info(f"🔄 Fetching data and analyzing **{ticker}** ...")

        with st.spinner("Processing sentiment and stock data..."):
            df, merged = sentiment_pipeline.process_sentiment(ticker, company)

            if not merged.empty:
                # --- Sentiment Trend Plot ---
                st.subheader("📊 Stock Sentiment Trend")
                fig = sentiment_pipeline.plot_stock_sentiment(merged, ticker)
                centered_fig_col = st.columns([1, 2, 1])
                with centered_fig_col[1]:
                    st.plotly_chart(fig, use_container_width=True)


                # --- Word Clouds in Row ---
                st.subheader("☁️ Sentiment Word Clouds")
                wc_cols = st.columns(3)
                for idx, sentiment_type in enumerate(["positive", "negative", "neutral"]):
                    wc = sentiment_pipeline.generate_wordcloud(df, sentiment_type)
                    with wc_cols[idx]:
                        if wc:
                            st.markdown(f"<h5 style='text-align: center;'>{sentiment_type.capitalize()} Word Cloud</h5>", unsafe_allow_html=True)
                            fig_wc, ax_wc = plt.subplots(figsize=(4, 3))
                            ax_wc.imshow(wc, interpolation="bilinear")
                            ax_wc.axis("off")
                            st.pyplot(fig_wc, use_container_width=True)
                        else:
                            st.info(f"No {sentiment_type} words found for {ticker}.")

                # --- Correlation Heatmap Centered ---
                st.subheader("📈 Sentiment vs Stock Price Correlation")
                fig_corr = sentiment_pipeline.plot_correlation_heatmap(merged, ticker)
                heatmap_cols = st.columns([1, 2, 1])
                with heatmap_cols[1]:
                    st.pyplot(fig_corr, use_container_width=False)

                # --- Predictive Stock Movement ---
                st.markdown('<div class="big-font">🚀 Predictive Stock Movement (Next Day)</div>', unsafe_allow_html=True)

                pred, prob = predict.predict_next_movement(merged)
                movement = "⬆️ UP" if pred == 1 else "⬇️ DOWN"
                confidence = f"{prob * 100:.2f}%"

                col1, col2 = st.columns(2)
                col1.markdown(f"<h2>Prediction: {movement}</h2>", unsafe_allow_html=True)
                col2.markdown(f"<h2>Confidence: {confidence}</h2>", unsafe_allow_html=True)


                st.markdown('<div class="stCaption">Prediction uses historical price, sentiment, SMA, and RSI to estimate the next day\'s movement.</div>', unsafe_allow_html=True)


                st.markdown('<p style="font-size:20px;color:green;">✅ Analysis complete.</p>', unsafe_allow_html=True)


            else:
                st.warning("⚠️ No data available for this stock. Please try another.")

# ------------------------- Sector-Wise Sentiment Aggregation -------------------------
elif st.session_state.page == 'sector':
    st.title("📊 Sector-Wise Sentiment Aggregation")
    if st.button("⬅️ Back to Stock Sentiment Analyzer"):
        st.session_state.page = 'main'
        st.rerun()

    

    # --- Sector-wise Sentiment Aggregation Chart at Top ---
    st.subheader("📊 Overall Sector-Wise Sentiment Overview")

    with st.spinner("Calculating overall sector sentiment..."):
        df_sector_overall = sector_analysis.aggregate_sector_sentiment_overall()

        if not df_sector_overall.empty:
            import plotly.express as px

            fig_sector_overall = px.bar(
                df_sector_overall.melt(id_vars='Sector', var_name='Sentiment', value_name='Count'),
                x='Sector',
                y='Count',
                color='Sentiment',
                barmode='group',
                title='Sector-Wise Sentiment Distribution (7 Days)',
                template='plotly_white',
                height=500
            )
            st.plotly_chart(fig_sector_overall, use_container_width=True)

            st.markdown(
                "<div style='color: green; font-size: 18px; font-weight: 500;'>"
                "✅ Use this chart to quickly identify which sectors have higher positive sentiment and potential momentum."
                "</div>",
                unsafe_allow_html=True
            )
        else:
            st.warning("⚠️ Unable to retrieve overall sector sentiment currently. Please try again later.")

    background_image_path = "background.jpg"



    if os.path.exists(background_image_path):
        set_background(background_image_path)
    else:
        st.warning("⚠ Background image not found.")

    

    st.subheader("🔍 Analyze Sentiment Across Companies")

    sector = st.selectbox("Select Sector for Aggregation", list(sectors.keys()))

    if st.button("Run Sector Analysis"):
        with st.spinner(f"Aggregating sentiment for **{sector}** sector..."):
            from analysis import sector_analysis

            df_sector_summary = sector_analysis.aggregate_sector_sentiment(sector)

            if not df_sector_summary.empty:
                st.subheader(f"✅ Sentiment Summary for {sector} Sector")
                st.dataframe(df_sector_summary, use_container_width=True, height=400)

                st.subheader("📊 Sentiment Comparison Across Companies")
                fig = px.bar(df_sector_summary, x='Company', y=['Positive', 'Negative', 'Neutral'], barmode='group', height=400, title='Sentiment Comparison Across Companies', template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    """
                    <div style= padding: 10px; border-radius: 5px">
                        <p style="color: #155724; font-size: 20px; font-weight: 500; margin: 0;">
                            Sector analysis complete. The company with the highest positive sentiment may be a candidate for further analysis.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning("⚠️ No sentiment data available for this sector currently.")
