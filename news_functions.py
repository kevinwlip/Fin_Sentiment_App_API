import pandas as pd
import datetime
import urllib.request
from bs4 import BeautifulSoup
from random import sample
from model_api import *

num_of_headlines = 20

# Print Timestamp At time of crawl
def news_date():
   news_date = str(datetime.date.today().strftime('%A, %B %d, %Y'))
   return news_date

# Get page and parse its contents
def news_scraper():
   url = "https://news.google.com/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US%3Aen"
   page = urllib.request.urlopen(url)
   soup = BeautifulSoup(page, "html.parser")

   try:
      business_heading = soup.find('title')
      if 'Business' in business_heading.get_text():
         print("Google News - Business Section Found.")
   except:
      raise Exception("Was not able to navigate to the Google News - Business section.")

   a_tags = soup.find_all('a')
   headlines = [tag.get_text() for tag in a_tags if len(tag.get_text().split()) >= 5]
   sample_headlines = sample(headlines, num_of_headlines)
   print(sample_headlines)
   return sample_headlines

# Format the output
def output_format(sample_headlines = news_scraper()):

   business_headlines = sample_headlines
   business_headlines = [f"{i}. {headline}" for i, headline in enumerate(business_headlines, start=1)]

   model1_results = []
   model2_results = []
   model3_results = []
   sentiment_models = ['kip', 'distilroberta', 'finbert']
   for i, model in enumerate(sentiment_models):
      for j, headline in enumerate(business_headlines):
         if i == 0:
               model1_results.append(query(model=sentiment_models[i], payload=headline))
         elif i == 1:
               model2_results.append(query(model=sentiment_models[i], payload=headline))
         elif i == 2:
               model3_results.append(query(model=sentiment_models[i], payload=headline))

   model1_vals = [item.values() for result in model1_results for item in result]
   model1_output = [f"{x.capitalize()},{y:.1%}" for x,y in model1_vals if x ]
   model1_output = [s.replace("Label_0", "Negative").replace("Label_1", "Positive").replace("Label_2", "Neutral") for s in model1_output]
   model1_sentiment = [output.split(',')[0] for output in model1_output]
   model1_probability = [output.split(',')[1] for output in model1_output]

   model2_vals = [item.values() for result in model2_results for item in result]
   model2_output = [f"{x.capitalize()},{y:.1%}" for x,y in model2_vals]
   model2_sentiment = [output.split(',')[0] for output in model2_output]
   model2_probability = [output.split(',')[1] for output in model2_output]

   model3_vals = [item.values() for result in model3_results for item in result]
   model3_output = [f"{x.capitalize()},{y:.1%}" for x,y in model3_vals]
   model3_sentiment = [output.split(',')[0] for output in model3_output]
   model3_probability = [output.split(',')[1] for output in model3_output]

   output = {"Kip Model Sentiments":model1_sentiment,
           "Kip Model Probabilities":model1_probability,
           "DistilRoberta Model Sentiments":model2_sentiment,
           "DistilRoberta Model Probabilities":model2_probability,
           "Finbert Model Sentiments":model3_sentiment,
           "Finbert Model Probabilities":model3_probability}

   df = pd.DataFrame(data=output, index=business_headlines)
   df.index.name = 'Business Headlines'

   return df, model1_sentiment, model1_probability, model2_sentiment, model2_probability, model3_sentiment, model3_probability
