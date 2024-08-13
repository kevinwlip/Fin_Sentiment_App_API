
# ## Source: https://huggingface.co/blog/sentiment-analysis-python
# 
# ## Note: A Couple Hiccups
# 0. Issue occurs where you have to 'Run All' twice due to 'accelerate' library not working, restart and run again
# 
# 1. Need the data from my Google Drive below and upload the .csv to session storage in Colab
# https://drive.google.com/file/d/1GtHhmDXqx9bcXVJvPmGuJFQSdpoYmWv1/view?usp=sharing
# 
# 2. Need to use your accesss token for HuggingFace
# 
# 3. Issue with the Trainer() cell? Try running again with 'Restart Session and Run All'


# 
# 
# # 0. Using Pre-trained Sentiment Analysis Models with Python


#%pip install torch
#%pip install pytorch-accelerated

#%pip install accelerate -U
#%pip install transformers -U


# There are more than 215 sentiment analysis models publicly available on the Hub and integrating them with Python just takes 5 lines of code
from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
data = ["I love you", "I hate you"]
sentiment_pipeline(data)


# Use a specific sentiment analysis model that is better suited to your language or use case by providing the name of the model.
# For example, if you want a sentiment analysis model for tweets, you can specify the model id
specific_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
specific_model(data)


# # 1. Activate GPU and Install Dependencies


# Activate GPU for faster training by clicking on 'Runtime' > 'Change runtime type' and then selecting GPU as the Hardware accelerator
# Then check if GPU is available
import torch
torch.cuda.is_available()


# Install required libraries
#%pip install datasets transformers huggingface_hub
#%apt-get install git-lfs


# #2. Preprocess data


from google.colab import files
uploaded = files.upload()


# Load data
from datasets import Dataset, load_dataset
from google.colab import files
import pandas as pd

# Example IMDB Dataset
#imdb = load_dataset("imdb")
#print("IMDB\n", imdb)
#print(imdb['train'][0])

# GUI for File Upload
#uploaded = files.upload()

# Processing 'fin_data.csv' from Section 16.4
#df = pd.read_csv("fin_data.csv")
#print("MAX", df['News Headline'].str.len().max())
#df['Label'] = df['Sentiment'].replace({'negative': 0, 'neutral': 1, 'positive': 2})
#del df['Unnamed: 0']
#df.to_csv('fin_data.csv', index=False)
#print(df)


fin = load_dataset("csv", data_files="fin_labeled_data.csv", split="train")
fin = fin.rename_column("News Headline", "text")
fin = fin.rename_column("Label", "label")
print("FIN\n", fin)

fin = fin.train_test_split(test_size=0.2)
print("FIN Train Test\n", fin)


# Create a smaller training dataset for faster training times
small_train_dataset = fin["train"].shuffle(seed=1).select([i for i in list(range(4000))])
small_test_dataset = fin["test"].shuffle(seed=1).select([i for i in list(range(1000))])
print("Train", small_train_dataset[0])
print("Test", small_test_dataset[0])


# Set DistilBERT tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# Prepare the text inputs for the model
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True)

tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
print(tokenized_train[0])
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)


# Use data_collector to convert our samples to PyTorch tensors and concatenate them with the correct amount of padding
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# # 3. Training the model


#%pip install evaluate


# Define DistilBERT as our base model:
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)


# Define the evaluation metrics
# Loss Functions - https://www.datacamp.com/tutorial/loss-function-in-machine-learning
# HuggingFace Loss Function Usage - https://stackoverflow.com/questions/71581197/what-is-the-loss-function-used-in-trainer-from-the-transformers-library-of-huggi
import numpy as np
import evaluate


def compute_metrics(eval_pred):
    load_accuracy = evaluate.load("accuracy")
    load_f1 =  evaluate.load("f1")
    load_mse =  evaluate.load("mse")
    load_mae =  evaluate.load("mae")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels, average='micro')["f1"]
    mse = load_mse.compute(predictions=predictions, references=labels, squared=True)["mse"]
    mae = load_mae.compute(predictions=predictions, references=labels)["mae"]
    return {"accuracy": accuracy, "f1": f1, "mse": mse, "mae": mae}


# Log in to your Hugging Face account
# Get your API token here https://huggingface.co/settings/token
from huggingface_hub import notebook_login

notebook_login()


#%pip install accelerate -U
#%pip install transformers -U


# Define a new Trainer with all the objects we constructed so far

from torch import nn
from transformers import TrainingArguments, Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0]).to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

repo_name = "financial-sentiment-model-5000-samples"

training_args = TrainingArguments(
    output_dir=repo_name,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,   # 5 here and 5 later in Step 6.
    weight_decay=0.01,
    save_strategy="epoch",
    push_to_hub=True,
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# Train the model
trainer.train()


# Compute the evaluation metrics
trainer.evaluate()


# Upload the model to the Hub
trainer.push_to_hub()


# # 4. Analyzing new data with multiple models - Self Sentiment Model, ProsusAI Finbert Model, and DistilRoberta Financial Sentiment Model


# Run inferences with your new model using Pipeline - https://huggingface.co/kevinwlip/financial-sentiment-model-5000-samples
from transformers import pipeline

sentiment_model = pipeline(model="kevinwlip/financial-sentiment-model-5000-samples")

sentiment_model(["I love this movie", "This movie sucks!"])

sentiment_model(["Energy shares drag on Wall Street as crude prices fall", "S&P 500 headed for a 67% downturn?", "Foreign investors navigate turmoil in Chinese markets with new playbook", "Fed Liquidity Drain Spoils Virus-Surge-Inspired Stock Buying-Panic", "Earnings per share ( EPS ) amounted to a loss of EUR0 .05",
                 "Equity markets likely to shine as Rupee's depreciation slows", "Investors cheered the blockbuster November jobs report", "Suven Life Sciences Q1 net up 16.68% at Rs 34.74 cr", "Airlines in sweet spot; expect SpiceJet to see rerating on IndiGo listing", "Twitter stock price target raised to $36.50 from $34.50 at Wedbush",
                 "A company is based in the district of the city", "The company did not disclose the price of the acquisition", "The 5 Best Finance Stocks of the Decade", "WeWork names new executives, path to profitability by 2023", "Chipotle highlights fast drive-thru lanes"])


# Try using the latest ProsusAI Finbert Model - https://huggingface.co/ProsusAI/finbert
from transformers import pipeline

finbert_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")

finbert_model(["I love this movie", "This movie sucks!"])

finbert_model(["Energy shares drag on Wall Street as crude prices fall", "S&P 500 headed for a 67% downturn?", "Foreign investors navigate turmoil in Chinese markets with new playbook", "Fed Liquidity Drain Spoils Virus-Surge-Inspired Stock Buying-Panic", "Earnings per share ( EPS ) amounted to a loss of EUR0 .05",
                 "Equity markets likely to shine as Rupee's depreciation slows", "Investors cheered the blockbuster November jobs report", "Suven Life Sciences Q1 net up 16.68% at Rs 34.74 cr", "Airlines in sweet spot; expect SpiceJet to see rerating on IndiGo listing", "Twitter stock price target raised to $36.50 from $34.50 at Wedbush",
                 "A company is based in the district of the city", "The company did not disclose the price of the acquisition", "The 5 Best Finance Stocks of the Decade", "WeWork names new executives, path to profitability by 2023", "Chipotle highlights fast drive-thru lanes"])


# Try using the latest DistilRoberta Financial Sentiment Model - https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis

from transformers import pipeline

distilroberta_model = pipeline("sentiment-analysis", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

distilroberta_model(["I love this movie", "This movie sucks!"])

distilroberta_model(["Energy shares drag on Wall Street as crude prices fall", "S&P 500 headed for a 67% downturn?", "Foreign investors navigate turmoil in Chinese markets with new playbook", "Fed Liquidity Drain Spoils Virus-Surge-Inspired Stock Buying-Panic", "Earnings per share ( EPS ) amounted to a loss of EUR0 .05",
                 "Equity markets likely to shine as Rupee's depreciation slows", "Investors cheered the blockbuster November jobs report", "Suven Life Sciences Q1 net up 16.68% at Rs 34.74 cr", "Airlines in sweet spot; expect SpiceJet to see rerating on IndiGo listing", "Twitter stock price target raised to $36.50 from $34.50 at Wedbush",
                 "A company is based in the district of the city", "The company did not disclose the price of the acquisition", "The 5 Best Finance Stocks of the Decade", "WeWork names new executives, path to profitability by 2023", "Chipotle highlights fast drive-thru lanes"])


# # 5. Fine-Tune Financial Sentiment Model 5000 - Huggingface


# Fine-Tune a Pre-Trained Model - https://huggingface.co/docs/transformers/en/training
# financial-sentiment-model-5000-samples Model

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

import numpy as np
import evaluate

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("kevinwlip/financial-sentiment-model-5000-samples-fine-tune")
# Define the mappings as dictionaries - Original Mappings: ("0": "Negative"), ("1": "Positive"), ("2": "Neutral")
id2label = {"0": "Negative", "1": "Positive", "2": "Neutral"}
label2id = {"Negative": "0", "Positive": "1", "Neutral": "2"}
# Define model checkpoint - can be the same model that you already have on the hub
model_ckpt = "kevinwlip/financial-sentiment-model-5000-samples-fine-tune"
config = AutoConfig.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=config)

# Tokenize the input data
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True) # Tokenize the 'text' column

# Apply the preprocessing to the datasets
tokenized_train_dataset = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = small_test_dataset.map(preprocess_function, batched=True)

# Training arguments
repo_name = "kevinwlip/financial-sentiment-model-5000-samples-fine-tune"

training_args = TrainingArguments(
    output_dir=repo_name,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,   # 5 from Step 3 and 5 here
    weight_decay=0.01,
    save_strategy="epoch",
    push_to_hub=True,
)

# Metric
load_accuracy = evaluate.load("accuracy")
load_f1 =  evaluate.load("f1")
load_mse =  evaluate.load("mse")
load_mae =  evaluate.load("mae")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels, average='micro')["f1"]
    mse = load_mse.compute(predictions=predictions, references=labels, squared=True)["mse"]
    mae = load_mae.compute(predictions=predictions, references=labels)["mae"]
    return {"accuracy": accuracy, "f1": f1, "mse": mse, "mae": mae}

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_train_dataset, # Use the tokenized training dataset
    eval_dataset=tokenized_test_dataset, # Use the tokenized test dataset
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train()


trainer.evaluate()


trainer.push_to_hub()


from transformers import pipeline

sentiment_model_tuned = pipeline("sentiment-analysis", model="kevinwlip/financial-sentiment-model-5000-samples-fine-tune")

""" Trying Text Data
sentiment_model_tuned(["I love this movie", "This movie sucks!"])

sentiment_model_tuned(["Energy shares drag on Wall Street as crude prices fall", "S&P 500 headed for a 67% downturn?", "Foreign investors navigate turmoil in Chinese markets with new playbook", "Fed Liquidity Drain Spoils Virus-Surge-Inspired Stock Buying-Panic", "Earnings per share ( EPS ) amounted to a loss of EUR0 .05",
                 "Equity markets likely to shine as Rupee's depreciation slows", "Investors cheered the blockbuster November jobs report", "Suven Life Sciences Q1 net up 16.68% at Rs 34.74 cr", "Airlines in sweet spot; expect SpiceJet to see rerating on IndiGo listing", "Twitter stock price target raised to $36.50 from $34.50 at Wedbush",
                 "A company is based in the district of the city", "The company did not disclose the price of the acquisition", "The 5 Best Finance Stocks of the Decade", "WeWork names new executives, path to profitability by 2023", "Chipotle highlights fast drive-thru lanes"])
"""


# # 6. Fine-Tune Pre-Existing ProsusAI Finbert Sentiment Model - Huggingface
# 


# ProsusAI Finbert Model

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

import numpy as np
import evaluate

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
# Define the mappings as dictionaries - Original Mappings: ("0": "Positive"), ("1": "Negative"), ("2": "Neutral")
id2label = {"0": "Positive", "1": "Negative", "2": "Neutral"}
label2id = {"Positive": "0", "Negative": "1", "Neutral": "2"}
# Define model checkpoint - can be the same model that you already have on the hub
model_ckpt = "ProsusAI/finbert"
config = AutoConfig.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=config)

# Tokenize the input data
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True) # Tokenize the 'text' column

# Apply the preprocessing to the datasets
tokenized_train_dataset = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = small_test_dataset.map(preprocess_function, batched=True)

# Training arguments
repo_name = "kevinwlip/ProsusAI-finbert-5000-samples-fine-tune"

training_args = TrainingArguments(
    output_dir=repo_name,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,   # 10 Total double self-trained model (Step 3. and Step 5.) which was split in two parts
    weight_decay=0.01,
    save_strategy="epoch",
    push_to_hub=True,
)

# Metric
load_accuracy = evaluate.load("accuracy")
load_f1 =  evaluate.load("f1")
load_mse =  evaluate.load("mse")
load_mae =  evaluate.load("mae")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels, average='micro')["f1"]
    mse = load_mse.compute(predictions=predictions, references=labels, squared=True)["mse"]
    mae = load_mae.compute(predictions=predictions, references=labels)["mae"]
    return {"accuracy": accuracy, "f1": f1, "mse": mse, "mae": mae}

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_train_dataset, # Use the tokenized training dataset
    eval_dataset=tokenized_test_dataset, # Use the tokenized test dataset
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train()


trainer.evaluate()


trainer.push_to_hub()


from transformers import pipeline

finbert_model_tuned = pipeline("sentiment-analysis", model="kevinwlip/ProsusAI-finbert-5000-samples-fine-tune")

""" Trying Text Data
finbert_model_tuned(["I love this movie", "This movie sucks!"])

finbert_model_tuned(["Energy shares drag on Wall Street as crude prices fall", "S&P 500 headed for a 67% downturn?", "Foreign investors navigate turmoil in Chinese markets with new playbook", "Fed Liquidity Drain Spoils Virus-Surge-Inspired Stock Buying-Panic", "Earnings per share ( EPS ) amounted to a loss of EUR0 .05",
                 "Equity markets likely to shine as Rupee's depreciation slows", "Investors cheered the blockbuster November jobs report", "Suven Life Sciences Q1 net up 16.68% at Rs 34.74 cr", "Airlines in sweet spot; expect SpiceJet to see rerating on IndiGo listing", "Twitter stock price target raised to $36.50 from $34.50 at Wedbush",
                 "A company is based in the district of the city", "The company did not disclose the price of the acquisition", "The 5 Best Finance Stocks of the Decade", "WeWork names new executives, path to profitability by 2023", "Chipotle highlights fast drive-thru lanes"])
"""


# # 7. Fine-Tune Pre-Existing DistilRoberta Financial Sentiment Model - Huggingface
# 
# 


# DistilRoberta Financial Sentiment Model

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

import numpy as np
import evaluate

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
# Define the mappings as dictionaries - Original Mappings: ("0": "Negative"), ("1": "Neutral"), ("2": "Positive")
id2label = {"0": "Negative", "1": "Neutral", "2": "Positive"}
label2id = {"Negative": "0", "Neutral": "1", "Positive": "2"}
# Define model checkpoint - can be the same model that you already have on the hub
model_ckpt = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
config = AutoConfig.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=config)

# Tokenize the input data
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True) # Tokenize the 'text' column

# Apply the preprocessing to the datasets
tokenized_train_dataset = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = small_test_dataset.map(preprocess_function, batched=True)

# Training arguments
repo_name = "kevinwlip/distilroberta-financial-sentiment-model-5000-samples-fine-tune"

training_args = TrainingArguments(
    output_dir=repo_name,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,   # 10 Total double self-trained model (Step 3. and Step 5.) which was split in two parts
    weight_decay=0.01,
    save_strategy="epoch",
    push_to_hub=True,
)

# Metric
load_accuracy = evaluate.load("accuracy")
load_f1 =  evaluate.load("f1")
load_mse =  evaluate.load("mse")
load_mae =  evaluate.load("mae")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels, average='micro')["f1"]
    mse = load_mse.compute(predictions=predictions, references=labels, squared=True)["mse"]
    mae = load_mae.compute(predictions=predictions, references=labels)["mae"]
    return {"accuracy": accuracy, "f1": f1, "mse": mse, "mae": mae}

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_train_dataset, # Use the tokenized training dataset
    eval_dataset=tokenized_test_dataset, # Use the tokenized test dataset
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train()


trainer.evaluate()


trainer.push_to_hub()


from transformers import pipeline

distilroberta_model_tuned = pipeline("sentiment-analysis", model="kevinwlip/distilroberta-financial-sentiment-model-5000-samples-fine-tune")

""" Trying Text Data
distilroberta_model_tuned(["I love this movie", "This movie sucks!"])

distilroberta_model_tuned(["Energy shares drag on Wall Street as crude prices fall", "S&P 500 headed for a 67% downturn?", "Foreign investors navigate turmoil in Chinese markets with new playbook", "Fed Liquidity Drain Spoils Virus-Surge-Inspired Stock Buying-Panic", "Earnings per share ( EPS ) amounted to a loss of EUR0 .05",
                 "Equity markets likely to shine as Rupee's depreciation slows", "Investors cheered the blockbuster November jobs report", "Suven Life Sciences Q1 net up 16.68% at Rs 34.74 cr", "Airlines in sweet spot; expect SpiceJet to see rerating on IndiGo listing", "Twitter stock price target raised to $36.50 from $34.50 at Wedbush",
                 "A company is based in the district of the city", "The company did not disclose the price of the acquisition", "The 5 Best Finance Stocks of the Decade", "WeWork names new executives, path to profitability by 2023", "Chipotle highlights fast drive-thru lanes"])
"""


# # 8. Trying Test Data


small_test_dataset[:]


# sentiment_model_tuned, using 1000 Test Data


smt = sentiment_model_tuned(small_test_dataset[:]['text'])
smt


smt_labels = [0 if item['label'] == 'Negative' else 1 if item['label'] == 'Positive' else 2 for item in smt[:]]


smt_matches = [x for x,y in zip(small_test_dataset[:]['label'], smt_labels) if x == y]
len(smt_matches)


# finbert_model_tuned, using 1000 Test Data


fmt = finbert_model_tuned(small_test_dataset[:]['text'])
fmt


fmt_labels = [0 if item['label'] == 'Positive' else 1 if item['label'] == 'Negative' else 2 for item in fmt[:]]


fmt_matches = [x for x,y in zip(small_test_dataset[:]['label'], fmt_labels) if x == y]
len(fmt_matches)


# distilroberta_model_tuned, using 1000 Test Data


dmt = distilroberta_model_tuned(small_test_dataset[:]['text'])
dmt


dmt_labels = [0 if item['label'] == 'Negative' else 1 if item['label'] == 'Neutral' else 2 for item in dmt[:]]


dmt_matches = [x for x,y in zip(small_test_dataset[:]['label'], dmt_labels) if x == y]
len(dmt_matches)


# Graph - Accuracy


import matplotlib.pyplot as plt

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')

Models = ['Own Sentiment Model', 'Finbert Model', 'DistilRoberta Model']
Accuracy = [round(len(smt_matches)/len(smt), 4)*100, round(len(fmt_matches)/len(fmt), 4)*100, round(len(dmt_matches)/len(dmt), 4)*100]

bars = plt.bar(Models, Accuracy)
bars[0].set_color('green')
bars[1].set_color('blue')
bars[2].set_color('red')

# giving title to the plot
plt.title("Model's Accuracy over 500 Text Sentiments")

# giving X and Y labels
plt.xlabel("Models")
plt.ylabel("Accuracy (in Percent %)")

addlabels(Models, Accuracy)

plt.show()


# # 9. Analysis
# 
# In Step 8, I attempted to do some fine-tuning on the three best performing models from Step 7 which were my Self Sentiment Model, the ProsusAI Finbert Model, and the DistilRoberta Financial Sentiment Model.
# 
# I trained on a 5000 sample dataset and it seemed the results of the fine-tuned models have reversed. My Self Sentiment Model performed the best, the Finbert Model second, and the DistilRoberta Model performed the worst still after updating the epochs to Self Sentiment Model Epoch=5, Then 5 again. The Finbert Model second and the DistilRoberta Model ran for 10 epochs each.
# 
# The 'eval_accuracy' for each of the models were very similar Self Sentiment Model: 0.815, the Finbert Model: 0.847, adn the DistilRoberta Model: 0.84.
# 
# However as shown in the graph, my Self Sentiment Model had a 81.5% accuracy, the Finbert Model had a 49.3% accuracy, and the DistilRoberta Model had a 24.5% accuracy which for an evenly distributed set of labels for test data is worse than randomly guessing which ideally gives us at 33.33% accuracy.
# 
# However, I still believe I will use the DistilRoberta Model overall for my Capstone due to its accuracy overall and try my Self Sentiment Model along with the Finbert Model just to see how it performs.


