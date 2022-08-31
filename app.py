import streamlit as st
from transformers import pipeline

@st.cache(allow_output_mutation=True)
def get_model():
    summarizer = pipeline("summarization", model="LongNN/TextSummarization")
    return summarizer


summarizer = get_model()

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")


if user_input and button :
    summary = summarizer(user_input,
                     num_return_sequences=5,
                     no_repeat_ngram_size=2,
                     early_stopping=True,
                     min_length=64,
                     max_length=256)
    st.write(summary[0]['summary_text'])
