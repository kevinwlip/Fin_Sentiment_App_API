
# ### Capstone 20.6 Step 5 - Data Wrangling & Exploration


import pandas as pd
import numpy as np
import json
from collections import Counter 
import matplotlib.pyplot as plt


# Datasets
# 
# https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news for 'financial_news.csv'
# 
# https://www.kaggle.com/datasets/ankurzing/aspect-based-sentiment-analysis-for-financial-news for 'financial_news_multiple_entity.csv'
# 
# https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment for 'twitter_financial_news_train.csv' and 'twitter_financial_news_valid.csv'


# Data Cleaning for 'financial_news.csv'

fin_news_df = pd.read_csv('financial_news.csv', header=None, encoding='latin-1')

fin_news_df.columns = ['Sentiment', 'News Headline']
fin_news_df_reindex = fin_news_df.reindex(columns=['News Headline', 'Sentiment'])
fin_news_df_reindex.index = np.arange(1, len(fin_news_df_reindex) + 1)
fin_news_df_final = fin_news_df_reindex

unique_sentiments = fin_news_df_final['Sentiment'].unique()
print(unique_sentiments)   # Only 3 values seen which is good - ['neutral', 'negative', 'positive']

print(fin_news_df_final)


# Data Cleaning for 'financial_news_multiple_entity.csv'

fin_news_mult_ent_df = pd.read_csv('financial_news_multiple_entity.csv')
fin_news_mult_ent_df = fin_news_mult_ent_df.drop('S No.', axis=1)
fin_news_mult_ent_df.columns = ['News Headline', 'Decisions', 'Words']
fin_news_mult_ent_df.index = np.arange(1, len(fin_news_mult_ent_df) + 1)

print("Original: \n")
print(fin_news_mult_ent_df.head(15))

# Use to compute Sentiment if *all* Sentiments are the same for multiple entities
def rev_dict(d):
    #return({v: k for k, v in json.loads(d).items()})
    new_dict = {v: k for k, v in json.loads(d).items()}
    length = len(new_dict.values())
    #return(new_dict, length)
    if length == 1:
        return(list(new_dict.keys())[0])
    else:
        return("")

fin_news_mult_ent_rd_applied_df = fin_news_mult_ent_df.copy(deep=True)

fin_news_mult_ent_rd_applied_df['Sentiment'] = fin_news_mult_ent_rd_applied_df['Decisions'].apply(rev_dict)

print("'rev_dict' Function Applied: \n")
print(fin_news_mult_ent_rd_applied_df.head(15))

'''
### Alternative Technique ###

dictionaries = fin_news_mult_ent_df_final['Decisions'].tolist()
#print(fin_news_mult_ent_df_final)

count = 0
sentiment_list = []
for text in dictionaries:
    #print(text)
    count += 1
    sentiments = re.findall(r'(neutral|negative|positive)', text)
    if sentiments:
        sentiment_set = set(sentiments)
        if len(sentiment_set) == 1:
            sentiment_list.append(sentiments)
            #print(sentiment_list)   # Will cause VSCode to crash
        else:
            sentiment_list.append('')

    #if count == 15:   # Loop ends after 15 rows
    #    break
print(sentiment_list)

fin_news_mult_ent_df_final['Sentiment'] = sentiment_list
fin_news_mult_ent_df_final['Sentiment'] = fin_news_mult_ent_df_final['Sentiment'].astype(str).str.strip("[]'")
'''

fin_news_mult_ent_dropped_df = fin_news_mult_ent_rd_applied_df.copy(deep=True)

fin_news_mult_ent_dropped_df['Sentiment'].replace('', np.nan, inplace=True)
fin_news_mult_ent_dropped_df.dropna(subset=['Sentiment'], inplace=True)
fin_news_mult_ent_df_final = fin_news_mult_ent_dropped_df

print("Dropped Empty Rows, Final: \n")
print(fin_news_mult_ent_df_final.head(15))


from collections import Counter 

fin_news_mult_ent_df2 = fin_news_mult_ent_df.copy(deep=True)

print("No operations applied: \n")
print(fin_news_mult_ent_df2.head(15))

# Use to compute Sentiment if *most* Sentiments are the same for multiple entities
def most_common(d):
    counts = Counter(json.loads(d).values())
    dict_counts = dict(counts)
    max_value = max(dict_counts.values())
    max_values = [k for k,v in dict_counts.items() if v == max_value]
    if len(max_values) == 1:
        return max_values[0]
    else:
        return("")

fin_news_mult_ent_mc_applied_df2 = fin_news_mult_ent_df2.copy(deep=True)

fin_news_mult_ent_mc_applied_df2['Sentiment'] = fin_news_mult_ent_mc_applied_df2['Decisions'].apply(most_common)

print("'most_common' Function Applied on New Column 'Sentiment2': \n")
print(fin_news_mult_ent_mc_applied_df2.head(15))

fin_news_mult_ent_dropped_df2 = fin_news_mult_ent_mc_applied_df2.copy(deep=True)

fin_news_mult_ent_dropped_df2['Sentiment'].replace('', np.nan, inplace=True)
fin_news_mult_ent_dropped_df2.dropna(subset=['Sentiment'], inplace=True)
fin_news_mult_ent_df2_final = fin_news_mult_ent_dropped_df2

print("Dropped Empty Rows, Final: \n")
print(fin_news_mult_ent_df2_final.head(15))


# Data Cleaning for 'twitter_financial_news_train.csv'

twitter_fin_news_train_df = pd.read_csv('twitter_financial_news_train.csv')
twitter_fin_news_train_df.columns = ['News Headline', 'Label']
twitter_fin_news_train_df_final = twitter_fin_news_train_df

conditions = [
    twitter_fin_news_train_df_final['Label'] == 0,
    twitter_fin_news_train_df_final['Label'] == 1,
    twitter_fin_news_train_df_final['Label'] == 2
]

categories = ['negative', 'positive', 'neutral']

twitter_fin_news_train_df_final['Sentiment'] = np.select(conditions, categories, default='')

print(twitter_fin_news_train_df_final)


# Data Cleaning for 'twitter_financial_news_valid.csv'

twitter_fin_news_valid_df = pd.read_csv('twitter_financial_news_valid.csv')
twitter_fin_news_valid_df.columns = ['News Headline', 'Label']
twitter_fin_news_valid_df_final = twitter_fin_news_valid_df

conditions = [
    twitter_fin_news_valid_df_final['Label'] == 0,
    twitter_fin_news_valid_df_final['Label'] == 1,
    twitter_fin_news_valid_df_final['Label'] == 2
]

categories = ['negative', 'positive', 'neutral']

twitter_fin_news_valid_df_final['Sentiment'] = np.select(conditions, categories, default='')

print(twitter_fin_news_valid_df_final)


# ##### 1. Creating the sentiment data, 'fin_data.csv' with 'rev_dict' applied on multiple entities file data, all sentiments have to be the same.


frames = [fin_news_df_final, fin_news_mult_ent_df_final, twitter_fin_news_train_df_final, twitter_fin_news_valid_df_final]
fin_data_df = pd.concat(frames)
fin_data_df = fin_data_df.drop(columns=['Decisions', 'Words', 'Label'])

fin_data_df = fin_data_df.drop_duplicates(subset=['News Headline'])

unique_sentiments = fin_data_df['Sentiment'].unique()
print(unique_sentiments)   # Only 3 values seen which is good - ['neutral', 'negative', 'positive']

fin_data_df.to_csv('fin_data.csv', index=False)
print(fin_data_df)


# ##### Creating the labeled data, 'fin_labeled_data.csv'


fin_labeled_data_df = pd.read_csv("fin_data.csv")
fin_labeled_data_df['Label'] = fin_labeled_data_df['Sentiment'].replace({'negative': 0, 'positive': 1, 'neutral': 2})
del fin_labeled_data_df['Sentiment']
fin_labeled_data_df.to_csv('fin_labeled_data.csv', index=False)
print(fin_labeled_data_df)


# The following steps are similar to the Mini-project on Exploratory Data Analysis (EDA)

fin_data_df.describe()


# Plot a Line Graph

x = fin_data_df['Sentiment'].unique()
y = fin_data_df['Sentiment'].value_counts()

print(x)
print(y)
plt.plot(x, y)


# ##### Create subset of the unlabeled data with 'rev_dict' applied - fin_data_df with 5000 negative, 5000 positive, and 5000 neutral random values, 'fin_data_subset.csv'


data_subset_neg_df = fin_data_df.query('(Sentiment == "negative")').sample(n=5000, random_state=0)
data_subset_pos_df = fin_data_df.query('(Sentiment == "positive")').sample(n=5000, random_state=0)
data_subset_neu_df = fin_data_df.query('(Sentiment == "neutral")').sample(n=5000, random_state=0)

subset_frames = [data_subset_neg_df, data_subset_pos_df, data_subset_neu_df]
fin_data_subset_df = pd.concat(subset_frames)
fin_data_subset_df = fin_data_subset_df.sample(frac=1, random_state=0)

fin_data_subset_df.to_csv('fin_data_subset.csv', index=False)
print(fin_data_subset_df)


# ##### Creating the labeled data subset, 'fin_labeled_data_subset.csv'


fin_labeled_data_subset_df = pd.read_csv("fin_data_subset.csv")
fin_labeled_data_subset_df['Label'] = fin_labeled_data_subset_df['Sentiment'].replace({'negative': 0, 'positive': 1, 'neutral': 2})
del fin_labeled_data_subset_df['Sentiment']
fin_labeled_data_subset_df.to_csv('fin_labeled_data_subset.csv', index=False)
print(fin_labeled_data_subset_df)


# ##### Create small subset of the unlabeled data with 'rev_dict' applied - fin_data_df with 500 negative, 500 positive, and 500 neutral random values, 'fin_data_small_subset.csv'


data_small_subset_neg_df = fin_data_df.query('(Sentiment == "negative")').sample(n=500, random_state=0)
data_small_subset_pos_df = fin_data_df.query('(Sentiment == "positive")').sample(n=500, random_state=0)
data_small_subset_neu_df = fin_data_df.query('(Sentiment == "neutral")').sample(n=500, random_state=0)

small_subset_frames = [data_small_subset_neg_df, data_small_subset_pos_df, data_small_subset_neu_df]
fin_data_small_subset_df = pd.concat(small_subset_frames)
fin_data_small_subset_df = fin_data_small_subset_df.sample(frac=1, random_state=0)

fin_data_small_subset_df.to_csv('fin_data_small_subset.csv', index=False)
print(fin_data_small_subset_df)


# ##### Creating the labeled data small subset, 'fin_labeled_data_small_subset.csv'


fin_labeled_data_small_subset_df = pd.read_csv("fin_data_small_subset.csv")
fin_labeled_data_small_subset_df['Label'] = fin_labeled_data_small_subset_df['Sentiment'].replace({'negative': 0, 'positive': 1, 'neutral': 2})
del fin_labeled_data_small_subset_df['Sentiment']
fin_labeled_data_small_subset_df.to_csv('fin_labeled_data_small_subset.csv', index=False)
print(fin_labeled_data_small_subset_df)


# ##### 2. Creating the sentiment data, 'fin_data2.csv' with 'most_common' applied on multiple entities file data, majority of sentiments have to be the same.


frames = [fin_news_df_final, fin_news_mult_ent_df2_final, twitter_fin_news_train_df_final, twitter_fin_news_valid_df_final]
fin_data_df2 = pd.concat(frames)
fin_data_df2 = fin_data_df2.drop(columns=['Decisions', 'Words', 'Label'])

fin_data_df2 = fin_data_df2.drop_duplicates(subset=['News Headline'])

unique_sentiments = fin_data_df2['Sentiment'].unique()
print(unique_sentiments)   # Only 3 values seen which is good - ['neutral', 'negative', 'positive']

fin_data_df2.to_csv('fin_data2.csv', index=False)
print(fin_data_df2)


# ##### Creating the labeled data, 'fin_labeled_data2.csv'


fin_labeled_data_df2 = pd.read_csv("fin_data2.csv")
fin_labeled_data_df2['Label'] = fin_labeled_data_df2['Sentiment'].replace({'negative': 0, 'positive': 1, 'neutral': 2})
del fin_labeled_data_df2['Sentiment']
fin_labeled_data_df2.to_csv('fin_labeled_data2.csv', index=False)
print(fin_labeled_data_df2)


# The following steps are similar to the Mini-project on Exploratory Data Analysis (EDA)

fin_data_df2.describe()


# Plot a Line Graph

x = fin_data_df2['Sentiment'].unique()
y = fin_data_df2['Sentiment'].value_counts()

print(x)
print(y)
plt.plot(x, y)


# ##### Create subset 2 of the unlabeled data with 'rev_dict' applied - fin_data_df with 5000 negative, 5000 positive, and 5000 neutral random values, 'fin_data_subset2.csv'


data_subset_neg_df2 = fin_data_df2.query('(Sentiment == "negative")').sample(n=5000, random_state=0)
data_subset_pos_df2 = fin_data_df2.query('(Sentiment == "positive")').sample(n=5000, random_state=0)
data_subset_neu_df2 = fin_data_df2.query('(Sentiment == "neutral")').sample(n=5000, random_state=0)

subset_frames2 = [data_subset_neg_df2, data_subset_pos_df2, data_subset_neu_df2]
fin_data_subset_df2 = pd.concat(subset_frames2)
fin_data_subset_df2 = fin_data_subset_df2.sample(frac=1, random_state=0)

fin_data_subset_df2.to_csv('fin_data_subset2.csv', index=False)
print(fin_data_subset_df2)


# ##### Creating the labeled data subset 2, 'fin_labeled_data_subset2.csv'


fin_labeled_data_subset_df2 = pd.read_csv("fin_data_subset2.csv")
fin_labeled_data_subset_df2['Label'] = fin_labeled_data_subset_df2['Sentiment'].replace({'negative': 0, 'positive': 1, 'neutral': 2})
del fin_labeled_data_subset_df2['Sentiment']
fin_labeled_data_subset_df2.to_csv('fin_labeled_data_subset2.csv', index=False)
print(fin_labeled_data_subset_df2)


# ##### Create small subset 2 of the unlabeled data with 'rev_dict' applied - fin_data_df with 500 negative, 500 positive, and 500 neutral random values, 'fin_data_small_subset2.csv'


data_small_subset_neg_df2 = fin_data_df2.query('(Sentiment == "negative")').sample(n=500, random_state=0)
data_small_subset_pos_df2 = fin_data_df2.query('(Sentiment == "positive")').sample(n=500, random_state=0)
data_small_subset_neu_df2 = fin_data_df2.query('(Sentiment == "neutral")').sample(n=500, random_state=0)

small_subset_frames2 = [data_small_subset_neg_df2, data_small_subset_pos_df2, data_small_subset_neu_df2]
fin_data_small_subset_df2 = pd.concat(small_subset_frames2)
fin_data_small_subset_df2 = fin_data_small_subset_df2.sample(frac=1, random_state=0)

fin_data_small_subset_df2.to_csv('fin_data_small_subset2.csv', index=False)
print(fin_data_small_subset_df2)


# ##### Creating the labeled data small subset 2, 'fin_labeled_data_small_subset2.csv'


fin_labeled_data_small_subset_df2 = pd.read_csv("fin_data_small_subset2.csv")
fin_labeled_data_small_subset_df2['Label'] = fin_labeled_data_small_subset_df2['Sentiment'].replace({'negative': 0, 'positive': 1, 'neutral': 2})
del fin_labeled_data_small_subset_df2['Sentiment']
fin_labeled_data_small_subset_df2.to_csv('fin_labeled_data_small_subset2.csv', index=False)
print(fin_labeled_data_small_subset_df2)

