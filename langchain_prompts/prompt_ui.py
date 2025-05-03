from dotenv import load_dotenv
import streamlit as st
import os

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatOpenAI(
    openai_api_key=groq_api_key,
    openai_api_base="https://api.groq.com/openai/v1",
    model_name="llama3-70b-8192"  # ✅ Updated model
)

st.header("Research Tool")

paper_input = st.selectbox("Select Reserarch Paper", ["Select...", "Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Beginnner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Length of Explanation", ["Short (1-2 paragraph)", "Medium (3-5 paragraphs)", "Long (detailed Explanation)"])

template = load_prompt("template.json")
# load the prompt template from the YAML file

if st.button("Summarize"):
    chain = template | model
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input,
    })
    st.write(result.content)   
