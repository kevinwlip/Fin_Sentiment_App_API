
README
======

This is a Financial Sentimental Prediction Application using HuggingFace Inference API (Paid, disabled now) hosted on Streamlit

Streamlit - [Financial Sentiment Analysis App](https://finsentimentappapi.streamlit.app/)

Machine Learning Models created by myself and are hosted on HuggingFace - https://huggingface.co/kevinwlip

GitHub - https://github.com/kevinwlip


Repository
-----------
- **'Data Work' folder** - contains Capstone Step 5: Data Wrangling Notebook with Financial Data (Input and Output)
- **'img' folder** - contains images used in the Application and the images of the Appication User Interface
- **'ML Models' folder** - contains Capstone Step 8: Scale Your Prototype Notebook with Fine-Tuned models uploaded to HuggingFace
- **'app.py'** - My Financial Sentiment Streamlit App
- **'news_functions.py'** - contains functions to parse Google News for use in 'app.py'
- **'requirements.txt'** - libraries need to run 'app.py'
- **'Docker files** - Dockerfile , .dockerignore , README.Docker.md , and compose.yaml will be created
- **'README.md'** - details about the project

Project User Interface
-----------------------

The Project is split into three sections:

1. Daily Headlines are parsed and sentiment analysis is run across three models - Kip, DistilRoberta, and Finbert
2. Distribution Graphs - Sentiment Distribution on Pie Chart & Probability Distribution using Line Graph.
3. Try Financial Sentiment Predictions - Select a model, input a business headline, and see whether you obtain Negative, Positive or Neutral

Examples of the Application are in the 'img' folder


Deploy Locally - Docker Containers
-----------------------------------

## First install Docker - https://www.docker.com/

### Two Ways to Deploy Locally

### Current Way: Using 'docker compose'

Originally created from the self-constructed Dockerfile using
```
$ docker init , console will prompt further questions to create a new Dockerfile
```
- New Dockerfile , .dockerignore , README.Docker.md , and compose.yaml will be created

Start the application by running:
```
$ docker compose up --build
```

Your application will be available at http://localhost:8501


### Other Way: Using self-constructed Dockerfile

Self-constructed Dockerfile contents:


```
FROM python:3.12

COPY . /fin_app

WORKDIR /fin_app

RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit","run"]

CMD ["app.py"]
```

Create the Docker Image:
```
$ docker build -t fin_image .
```

Run the Docker Image in a new container:
```
$ docker run -p 8501:8501 fin_image:latest
```


Cloud Deployment - Streamlit
-----------------------------

[Streamlit](https://streamlit.io/)


Deployment Alternatives - AWS EC2
---------------------------------

Sources:
1. [Deploying an OpenAI Streamlit Python App on AWS EC2](https://www.youtube.com/watch?v=oynd7Xv2i9Y)
2. [AWS Tutorials: Deploy Python Application on AWS EC2](https://www.youtube.com/watch?v=3sQhVKO5xAA)


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
$ git clone https://github.com/kevinwlip/Fin_Sentiment_App_API.git
$ cd [repo]
$ python3 -m pip install tensorflow --no-cache-dir , help prevent crash issue
- Crash Issue: https://stackoverflow.com/questions/67381812/tensorflow-installation-killed-on-aws-ec2-instance
$ python3 -m pip install --ignore-installed streamlit , prevents issue with requests library
$ python3 -m pip install -r requirements.txt
```

Run App

```
$ python3 -m streamlit run app.py
```

Keep app running, even if you close the EC2 instance window/terminal

```
$ nohup python3 -m streamlit run app.py
```

Look for process ID and kill the app, to prevent AWS charges
```
$ sudo su
$ ps -ef
$ kill [PID]
```
