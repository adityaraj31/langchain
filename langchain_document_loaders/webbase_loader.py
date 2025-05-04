from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama3-8b-8192"
)

prompt = PromptTemplate(
    template="Answer the following question \n {question} from the following text \n {text}",
    input_variables=["question", "text"]
)

parser = StrOutputParser()

url = 'https://blinkit.com/prn/aashirvaad-high-fibre-atta-with-multigrains-5-kg/prid/108301'
loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

question = "What is the product that we are talking about?"

output = chain.invoke({"question": question, "text": docs[0].page_content})

print(output)