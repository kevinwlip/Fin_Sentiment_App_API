import streamlit as st
import requests

@st.cache_data(show_spinner=False)
def query(model, payload):

    if model == 'kip':
        API_URL = "https://r6w5s9ero5zycv22.us-east-1.aws.endpoints.huggingface.cloud"
    elif model == 'distilroberta':
        API_URL = "https://api-inference.huggingface.co/models/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    elif model == 'finbert':
        API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"

    ### Please replace st.secrets['auth_token'] with "Bearer [HF_TOKEN]" if running locally, else you will see a 'secrets' error ###
    headers = {"Authorization": "Bearer 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"}
    
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": payload})
        output = response.json()

        if model == 'distilroberta' or model == 'finbert':
            output = [max(output[0], key=lambda x:x['score'])]
    
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)

    return output

#print(query(model='finbert', payload="Stocks fell really hard today."))
