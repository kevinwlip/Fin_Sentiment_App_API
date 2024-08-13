import streamlit as st
import requests

def query(model, payload):

    if model == 'kip':
        API_URL = "https://r6w5s9ero5zycv22.us-east-1.aws.endpoints.huggingface.cloud"
    elif model == 'distilroberta':
        API_URL = "https://api-inference.huggingface.co/models/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    elif model == 'finbert':
        API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"

    headers = {"Authorization": st.secrets["auth_token"]}

    response = requests.post(API_URL, headers=headers, json={"inputs": payload})
    output = response.json()

    if model == 'distilroberta' or model == 'finbert':
        output = [max(output[0], key=lambda x:x['score'])]

    return output

print(query(model='finbert', payload="Stocks fell really hard today."))
