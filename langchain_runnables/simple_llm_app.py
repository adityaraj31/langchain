from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model_name="llama3-8b-8192",
)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Suggest a catchy blog title about {topic}"
)

topic = input('Enter a topic')

formatted_prompt = prompt.format(topic=topic)

blog_title = llm.predict(formatted_prompt)

print("Generated Blog Title", blog_title)