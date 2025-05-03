from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

load_dotenv()

# Joke generation prompt
joke_prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

# Explanation prompt
explain_prompt = PromptTemplate(
    template="Explain the following joke: {joke}",
    input_variables=["joke"]
)

# Model and parser
model = ChatGroq(model="llama3-8b-8192")
parser = StrOutputParser()

# Joke generation chain
joke_chain = joke_prompt | model | parser

# Explanation chain (takes a joke as input)
explanation_chain = explain_prompt | model | parser

# Parallel chain using RunnablePassthrough:
# - `RunnablePassthrough()` passes the joke to 'joke'
# - The joke also flows into `explanation_chain`
parallel_chain = RunnablePassthrough() | RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation": explanation_chain
})

# Final chain: generate a joke, then run the parallel process on it
final_chain = joke_chain | parallel_chain

# Invoke with a topic
result = final_chain.invoke({"topic": "cats"})

print(result)
