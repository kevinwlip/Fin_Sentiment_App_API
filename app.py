import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from news_functions import *

st.set_page_config(page_title="Sentiment Analysis on News Headlines", layout="wide")

# Data Organization
@st.cache_data(show_spinner="Hi Users, Fetching Data.. Please Wait...")
def return_dfs():
    df, model1_sentiment, model1_probability, model2_sentiment, model2_probability, model3_sentiment, model3_probability = output_format()

    m1_s_counts = Counter(model1_sentiment)
    m2_s_counts = Counter(model2_sentiment)
    m3_s_counts = Counter(model3_sentiment)

    pie_data = {'Model': ['Kip', 'DistilRoberta', 'Finbert'],
                'Negative': [m1_s_counts['Negative'], m2_s_counts['Negative'], m3_s_counts['Negative']],
                'Positive': [m1_s_counts['Positive'], m2_s_counts['Positive'], m3_s_counts['Positive']],
                'Neutral': [m1_s_counts['Neutral'], m2_s_counts['Neutral'], m3_s_counts['Neutral']],
                'Sum': [m1_s_counts['Negative']+m1_s_counts['Positive']+m1_s_counts['Neutral'], 
                        m2_s_counts['Negative']+m2_s_counts['Positive']+m2_s_counts['Neutral'],
                        m3_s_counts['Negative']+m3_s_counts['Positive']+m3_s_counts['Neutral']]}

    model1_probability = [prob.strip('%') for prob in model1_probability]
    model2_probability = [prob.strip('%') for prob in model2_probability]
    model3_probability = [prob.strip('%') for prob in model3_probability]

    line_data = {'Num_Of_Headlines': list(range(num_of_headlines+1))[1:],
                'Kip': model1_probability,
                'DistilRoberta': model2_probability,
                'Finbert': model3_probability}

    pie_df = pd.DataFrame(pie_data)

    line_df = pd.DataFrame(line_data)

    return df, pie_df, line_df

df, pie_df, line_df = return_dfs()

icon, header = st.columns(spec=[1,40], vertical_alignment="center")
with icon:
    st.image(image="imgs/gnews.jpg", width=50)
with header:
    st.header(f"Financial Headlines for today: {news_date()}")

st.dataframe(df.style.map(lambda x: f"background-color: {'red' if x=='Negative' else 'green' if x=='Positive' else 'gray'}", 
                          subset=['Kip Model Sentiments', 'DistilRoberta Model Sentiments', 'Finbert Model Sentiments']), 
                          use_container_width=True)

st.header("Distribution Graphs of the Headlines")
with st.container(border=True):
    left_column, right_column = st.columns(2)

    pie, line = st.columns(spec=[1,1])
    with pie:
        sentiment_types = st.selectbox('Sentiment Type', ['Negative', 'Positive', 'Neutral'], label_visibility='collapsed')
        
        fig = px.pie(pie_df, values=sentiment_types, names='Model', title=f'Number of {sentiment_types} types', category_orders={"Model": ["Kip", "DistilRoberta", "Finbert"]}, height=300, width=200)
        fig.update_layout(margin=dict(l=100, r=0, t=30, b=0), font=dict(size=16), hoverlabel=dict(font_size=16), legend=dict(font=dict(size=16)))
        st.plotly_chart(fig, use_container_width=True)

    with line:
        fig = px.line(line_df, x=line_df['Num_Of_Headlines'], y=[line_df['Kip'], line_df['DistilRoberta'], line_df['Finbert']], title='Probability Distributions')
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=0), xaxis_title='Number of Headlines', yaxis_title='Probability (%)', legend_title_text='Models', legend_title=dict(font=dict(size=18)), legend=dict(font=dict(size=16)))
        fig.update_xaxes(tickfont=dict(size=14))
        fig.update_yaxes(tickfont=dict(size=14))
        st.plotly_chart(fig, use_container_width=True)

st.header("Try Financial Sentiment Predictions Yourself")
with st.container(border=True):
    left_column, right_column = st.columns(2)

    def predict():
        if model == "Kip's Self-Trained Model":
            sentiment_model = 'kip'
        elif model == "DistilRoberta":
            sentiment_model = 'distilroberta'
        elif model == "Prosus AI Finbert":
            sentiment_model = 'finbert'

        vals = [result.values() for result in query(model=sentiment_model, payload=input)]
        output = [f"{x.capitalize()}, Probability: {y:.1%}" for x,y in vals][0]
        output = output.replace("Label_0", "Negative").replace("Label_1", "Positive").replace("Label_2", "Neutral")

        return output

    with left_column:
        st.header("Select Your Model:")
        model = st.selectbox('Model', ["Kip's Self-Trained Model", "DistilRoberta", "Prosus AI Finbert"])
        st.caption("Trained using 4K Training Data and 1K Test Data")

    with right_column:
        st.header("Input Your Financial Text:")
        input = st.text_input(label = "Financial Text", value = "Sales have risen in the export markets")

        if st.button("Predict Sentiment"):
            output = predict()
            st.success(output, icon="🎉")
