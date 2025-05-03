from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()
 
prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

model = ChatGroq(
    model_name="llama3-8b-8192",
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="Expalin the following joke: {text}",
    input_variables=['text']
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

print(chain.invoke({'topic': 'basketball'}))

