import os

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")

llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
client = OpenAI(api_key=api_key)

title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""You are an experienced speech writer.
You need to craft an impactful title for a speech
on the following topic: {topic}
Answer exactly with one title.
"""
)

speech_prompt = PromptTemplate(
    input_variables=["title"],
    template="""You need to write a powerful speech of 350 words
for the following title: {title}.

The speech must naturally include all of these emotional qualities:
Inspiring, Passionate, Empathetic, Motivational, Heartfelt,
Uplifting, Empowering, Sincere, Hopeful, and Enthusiastic.

Make it sound natural, emotionally rich, and suitable for public speaking.
"""
)

def show_title(title):
    st.subheader("Generated Title")
    st.write(title)
    return title

first_chain = title_prompt | llm | StrOutputParser() | show_title
second_chain = speech_prompt | llm
final_chain = first_chain | second_chain

st.title("Speech Generator with Voice")

topic = st.text_input("Enter a topic")

voice = st.selectbox(
    "Choose a voice",
    [
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "fable",
        "nova",
        "onyx",
        "sage",
        "shimmer",
    ]
)

if st.button("Generate Speech and Audio"):
    if topic:
        response = final_chain.invoke({"topic": topic})
        speech_text = response.content

        st.subheader("Generated Speech")
        st.write(speech_text)

        audio_response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=speech_text,
            response_format="mp3"
        )

        st.subheader("Listen to the Speech")
        st.audio(audio_response.content, format="audio/mp3")
    else:
        st.warning("Please enter a topic.")