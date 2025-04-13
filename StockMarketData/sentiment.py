from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import os
import httpx
from dotenv import load_dotenv
import time

load_dotenv("passcodes.env")

API= os.getenv("API_KEY")

def get_news_sentiment(stock, start_date, end_date):
    url = f"https://newsapi.org/v2/everything?q={stock}&from={start_date}&to={end_date}&sortBy=popularity&apiKey={API}"
    with httpx.Client(http2=True) as client:
        response = client.get(url)
        
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.json()}")
        return None

    response = requests.get(url)
    articles = response.json().get('articles', [])
    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    article_urls = []
    sentiment_scores = []
    for article in articles:
        # Handle None values for title and description
        title = article.get('title') or ''
        description = article.get('description') or ''
        sentiment = sentiment_analyzer.polarity_scores(title + ' ' + description)
        url = article.get('url')  # Retrieve the article URL
        sentiment_scores.append(sentiment['compound'])
        if len(article_urls)<5:
            article_urls.append(url)  # Collect the URL of the article
    if sentiment_scores:
        return sum(sentiment_scores) / len(sentiment_scores), article_urls if sentiment_scores else None

# start_date = "2024-10-13"
# end_date = "2024-10-13"
# stock = "CLSK"

# sentiment_score = get_news_sentiment(stock,start_date, end_date)

# print(sentiment_score)

