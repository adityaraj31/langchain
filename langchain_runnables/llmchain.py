from langchain.llms import openai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = openai(model_name="gtp-3.5-turbo", temperature=0.7)

prompt = PromptTemplate(
    template="Suggest a catchy blog title about {topic}",
    input_variables=["topic"]
)

chain = LLMChain(llm=llm, prompt=prompt)

topic = input("Enter a topic")
output = chain.run(topic)

print("Genereated Blog Title: ", output)