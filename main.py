import pandas as pd
import streamlit as st
from helper import *

st.title("Video Sentiment Analysis")
st.markdown('''
<style>
        .translucent-text {
            opacity: 0.8; /* Adjust the opacity value as needed */
        }
    </style>
''', unsafe_allow_html=True)
st.write("""
<div class="translucent-text">
    Using <a href="https://huggingface.co/distil-whisper/distil-medium.en" target="_blank">DistilWhisper</a> (
    OpenAI/HuggingFace), <a href="https://huggingface.co/facebook/bart-large-cnn" target="_blank">BERT-Large</a> (
    Meta), 
    And <a href="https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment" target="_blank">RoBERTa</a> (
    cardiffnlp)
</div>
""", unsafe_allow_html=True)

st.write("""

        The first load might take allot of time since it will redownload the models, but after that it will be much
            faster. Remember that this runs on the CPU, so it might take a while to process the video.
         This app performs sentiment analysis on YouTube videos! Enter the link of the video and it will be 
         transcribed and analyzed. """)

link = st.text_input("Enter the link of the YouTube video", placeholder='https://www.youtube.com/watch?v=dQw4w9WgXcQ')
if link:
    # URL validation
    if not link.startswith("https://www.youtube.com/watch?v="):
        st.error("Please enter a valid YouTube link")
        st.stop()
    with st.spinner('Downloading video...'):
        try:
            path = download_yt_video(link=link)
        except Exception as e:
            st.error(e)
            st.stop()

    with st.spinner('Transcribing video...'):
        try:
            text = get_transcript(path=path)['text']
        except Exception as e:
            st.error(e)
            st.stop()

    with st.spinner('Summarizing video...'):
        try:
            summ = summery_video(text=text)[0]['summary_text']
            st.header("Summary of the Video")
            st.write(summ)
        except Exception as e:
            st.error(e)
            st.stop()

    with st.spinner('Analyzing video...'):
        try:
            sent = get_sentiment(text=summ)
            chart = st.bar_chart(pd.DataFrame(sent, index=['Bad', 'Neutral', 'Good']).T, color=['#ff5555', '#55ff55',
                                                                                                '#ffff55'])
            st.code(f"Bad: {sent[0]}\nNeutral: {sent[1]}\nGood: {sent[2]}", language='json')
        except Exception as e:
            st.error(e)
            st.stop()