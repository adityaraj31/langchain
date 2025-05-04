from langchain_community.document_loaders import WebBaseLoader

url = 'https://blinkit.com/prn/aashirvaad-high-fibre-atta-with-multigrains-5-kg/prid/108301'
loader = WebBaseLoader(url)

docs = loader.load()

print(len(docs))

print(docs[0].page_content)