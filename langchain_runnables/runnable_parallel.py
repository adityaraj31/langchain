from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_varaible=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a linkedIn post about {topic}',
    input_varaible=['topic']
)

model = ChatGroq(
    model="llama3-8b-8192"
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedIn': RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({"topic": "AI"})

print(result)