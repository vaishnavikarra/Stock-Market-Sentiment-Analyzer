# import pandas as pd
# from analysis import sentiment_pipeline
# from analysis.sectors import sectors


# def aggregate_sector_sentiment_overall():
#     sector_sentiment_list = []

#     for sector_name, companies in sectors.items():
#         pos_count = 0
#         neg_count = 0
#         neu_count = 0
#         total_posts = 0

#         for company_name, info in companies.items():
#             ticker = info["ticker"]

#             try:
#                 df, _ = sentiment_pipeline.process_sentiment(ticker, company_name, num_days=7)
#                 sentiment_counts = df['sentiment'].value_counts()
#                 pos_count += sentiment_counts.get('positive', 0)
#                 neg_count += sentiment_counts.get('negative', 0)
#                 neu_count += sentiment_counts.get('neutral', 0)
#                 total_posts += len(df)
#             except Exception as e:
#                 print(f"Error processing {ticker}: {e}")

#         if total_posts > 0:
#             sector_sentiment_list.append({
#                 'Sector': sector_name,
#                 'Positive': pos_count,
#                 'Negative': neg_count,
#                 'Neutral': neu_count
#             })

#     return pd.DataFrame(sector_sentiment_list)


# def aggregate_sector_sentiment(sector_name):
#     companies = sectors[sector_name]
#     sentiment_summary = []

#     for company_name, info in companies.items():
#         ticker = info["ticker"]
#         try:
#             df, merged = sentiment_pipeline.process_sentiment(ticker, company_name)
#             if not df.empty:
#                 counts = df["sentiment"].value_counts()
#                 pos = counts.get("positive", 0)
#                 neg = counts.get("negative", 0)
#                 neu = counts.get("neutral", 0)
#                 score = pos - neg

#                 sentiment_summary.append({
#                     "Company": company_name,
#                     "Ticker": ticker,
#                     "Positive": pos,
#                     "Negative": neg,
#                     "Neutral": neu,
#                     "Score": score
#                 })
#         except Exception as e:
#             print(f"Error processing {company_name}: {e}")

#     df_summary = pd.DataFrame(sentiment_summary)
#     if not df_summary.empty:
#         df_summary = df_summary.sort_values(by="Score", ascending=False)
#     return df_summary
import pandas as pd
from analysis.sectors import sectors
from analysis import sentiment_pipeline

# -------------------- 1️⃣ Overall Sector-Wise Sentiment Aggregation --------------------
def aggregate_sector_sentiment_overall():
    """
    Aggregates sentiment counts across all sectors over the last 7 days
    for a top-level momentum overview.
    """
    sector_sentiment_list = []

    for sector_name, companies in sectors.items():
        pos_count = 0
        neg_count = 0
        neu_count = 0
        total_posts = 0

        for company_name, info in companies.items():
            ticker = info["ticker"]

            try:
                df, _ = sentiment_pipeline.process_sentiment(ticker, company_name, num_days=7)
                sentiment_counts = df['sentiment'].value_counts()
                pos_count += sentiment_counts.get('positive', 0)
                neg_count += sentiment_counts.get('negative', 0)
                neu_count += sentiment_counts.get('neutral', 0)
                total_posts += len(df)
            except Exception as e:
                print(f"[WARN] Skipping {ticker} due to error: {e}")

        if total_posts > 0:
            sector_sentiment_list.append({
                'Sector': sector_name,
                'Positive': pos_count,
                'Negative': neg_count,
                'Neutral': neu_count
            })

    return pd.DataFrame(sector_sentiment_list)


# -------------------- 2️⃣ Within-Sector Company-Wise Sentiment Aggregation --------------------
def aggregate_sector_sentiment(sector_name):
    """
    Aggregates sentiment counts for each company within the specified sector
    over the last 7 days.
    """
    company_sentiment_list = []

    companies = sectors.get(sector_name, {})
    for company_name, info in companies.items():
        ticker = info["ticker"]

        try:
            df, _ = sentiment_pipeline.process_sentiment(ticker, company_name, num_days=7)
            sentiment_counts = df['sentiment'].value_counts()
            pos_count = sentiment_counts.get('positive', 0)
            neg_count = sentiment_counts.get('negative', 0)
            neu_count = sentiment_counts.get('neutral', 0)
            total_posts = len(df)

            if total_posts > 0:
                company_sentiment_list.append({
                    'Company': company_name,
                    'Positive': pos_count,
                    'Negative': neg_count,
                    'Neutral': neu_count
                })
        except Exception as e:
            print(f"[WARN] Skipping {ticker} due to error: {e}")

    return pd.DataFrame(company_sentiment_list)
