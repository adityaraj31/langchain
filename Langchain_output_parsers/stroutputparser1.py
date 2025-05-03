from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    model_name="llama3-8b-8192",  # or "mixtral-8x7b-32768", etc.
)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template="Write a detailed report on the following topic: {topic}",
    input_variables=["topic"],
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text: {text}",
    input_variables=["text"],
)

# 1st prompt run
prompt1 = template1.format(topic="black hole")
result = llm.invoke(prompt1)

# 2nd prompt run
prompt2 = template2.format(text=result.content)
result1 = llm.invoke(prompt2)

parser = StrOutputParser()

chain = tempalte1 | model | parser | template2 | model | parser 

result = chain.invoke({'topic': 'black hole'})

print(result)