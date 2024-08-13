
README
======

This is a Financial Sentimental Prediction Application created through 
Streamlit and hosted on Amazon EC2

Machine Learning Models created by myself and are hosted on HuggingFace

HuggingFace - https://huggingface.co/kevinwlip

GitHub - https://github.com/kevinwlip


Repository
----------
'Data Work' folder - contains Capstone Step 5: Data Wrangling Notebook with Financial Data (Input and Output)
'img' folder - contains Images for the App and the App UI
'ML Models' folder - contains Capstone Step 8: Scale Your Prototype Notebook with Fine-Tuned models uploaded to HuggingFace
'app.py' - Streamlit App
'news_functions.py' - contains functions to parse Google News for use in 'app.py'
'requirements.txt' - libraries need to run 'app.py'
'README.md' - details about the project


Project User Interface
----------------------
The Project is split into three sections:

1. Daily Headlines are parsed and sentiment analysis is run across three models - Kip, DistilRoberta, and Finbert
2. Distribution Graphs - Sentiment Distribution on Pie Chart & Probability Distribution using Line Graph.
3. Try Financial Sentiment Predictions - Select a model, input a business headline, and see whether you obtain Negative, Positive or Neutral


Deployment - Streamlit with AWS EC2, but currently deployed on Streamlit
------------------------------------------------------------------------

Streamlit - https://finsentimentapp.streamlit.app/

Sources:
1. (Main) Deploying an OpenAI Streamlit Python App on AWS EC2 - https://www.youtube.com/watch?v=oynd7Xv2i9Y
2. AWS Tutorials: Deploy Python Application on AWS EC2 - https://www.youtube.com/watch?v=3sQhVKO5xAA&t=141s


Have app contents on GitHub.

Connect to AWS EC2 Instance.

In EC2 command line:

```
$ sudo su
$ yum update
$ yum install git
$ yum install python3-pip
```

Get GitHub link to repo.
```
$ git clone https://github.com/kevinwlip/Fin_Sentiment_App.git
$ cd [repo]
$ python3 -m pip install tensorflow --no-cache-dir , help prevent crash issue
- Crash Issue: https://stackoverflow.com/questions/67381812/tensorflow-installation-killed-on-aws-ec2-instance
$ python3 -m pip install --ignore-installed streamlit , prevents issue with requests library
$ python3 -m pip install -r requirements.txt
```

Run App

`$ python3 -m streamlit run app.py`

Keep app running, even if you close the EC2 instance window/terminal

`$ nohup python3 -m streamlit run app.py`

Look for process ID and kill the app, to prevent AWS charges
```
$ sudo su
$ ps -ef
$ kill [PID]
```