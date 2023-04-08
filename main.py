from urllib.request import urlopen, Request

import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# urllib -> open, read and download urls
# BeautifulSoup - Web-scraping
# sentiment analysis identifies te emotional tone of a text
# panda helps apply functions (analyser) quickly onto the datasets
finviz_url = "https://finviz.com/quote.ashx?t="
tickers = ["AMZN", "GOOG", "FB", "AAPL"]

news_tables = {}

for ticker in tickers:
    url = finviz_url + ticker

    req = Request(url=url, headers={"user-agent": "my-app"})
    # headers and "user-agent"-> allow data access and download
    response = urlopen(req)

    html = BeautifulSoup(response, features="html.parser")
    # parse html file
    news_table = html.find(id="news-table")
    # search the html file and get entire table with the specified id
    news_tables[ticker] = news_table
    # find "news-table" for AMZN, AMD, FB
    # break

# amzn_data = news_tables["AMZN"]
# amzn_rows = amzn_data.findAll("tr") # give all HTML objects inside of "tr" elements

# enumerate -> output will be index row
# for index, row in enumerate(amzn_rows):
# title = row.a.text
# in the table row, find text (title) inside of <a> </a>
# timestamp = row.td.text
# print(timestamp + "  " + title)

parsed_data = []

for ticker, news_table in news_tables.items():

    for row in news_table.findAll("tr"):
        title = row.a.get_text()
        # in the table row, text within <a></a> is title/headline
        # .get_text() is the same as .text()
        date_data = row.td.text.split(" ")
        # (date time) split into (date), (time)

        # if the date only has one item
        if len(date_data) == 1:
            time = date_data[0]
            # the first item is time

        else:
            date = date_data[0]
            time = date_data[1]

        parsed_data.append([ticker, date, time, title])

df = pd.DataFrame(parsed_data, columns=["ticker", "date", "time", "title"])

vader = SentimentIntensityAnalyzer()

f = lambda title: vader.polarity_scores(title)["compound"]
df["compound"] = df["title"].apply(f)
# create a compound column and add f-score (compounds from sentiment analysis)
df["date"] = pd.to_datetime(df.date).dt.date

plt.figure(figsize=(10, 8))

mean_df = df.groupby(["ticker", "date"]).mean()
mean_df = mean_df.unstack()
# unstack then remove the name of compound by using .xs
mean_df = mean_df.xs("compound", axis="columns").transpose()
mean_df.plot(kind="bar")
plt.show()
