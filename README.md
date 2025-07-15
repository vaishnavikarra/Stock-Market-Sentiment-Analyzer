
The Stock Market Sentiment Analyzer is an application that uses data and financial data to gain insights into how public sentiment relates to stock price movements. The main goal is to identify sentiment related to a stock/company using the aggregate data from many different unstructured sources ie. news articles, Reddit, Twitter etc.
Once the data is collected sentiment classification models  can use classification methods to classify the content as positive, negative, or neutral. Once the data has been processed, to visualize the sentiment, data can be presented in time-series with time-based plots, word clouds, and correlation plots to help investors see the trends in public opinion over time and correlates with stock price. 

Tech stack:
Streamlit application
Python
feedparser,Praw
NLTK- preprocessing,tokenization
Vader sentiment analyzer
SMA,RSI 

Output:
Stock price Vs sentiment count graph
positive,negative,neutral word clouds
heat map (colrelation between close price,positive,negative,neutra sentiment count)
Next day prediction for tsock movement with confidence
Sector wise sentimnet aggregation- all sectors,company wise

Results :


<img width="828" height="364" alt="image" src="https://github.com/user-attachments/assets/61a2f20a-8582-4de4-9cca-3937a5c09a6c" />

<img width="819" height="380" alt="image" src="https://github.com/user-attachments/assets/5565da02-0672-458c-8b1d-a25b438b0b31" />

<img width="799" height="402" alt="image" src="https://github.com/user-attachments/assets/688e32d1-afee-4540-bb63-aa22d8f8f26a" />

<img width="802" height="399" alt="image" src="https://github.com/user-attachments/assets/d4fd06e6-c509-4627-a429-ac12a23e892f" />




