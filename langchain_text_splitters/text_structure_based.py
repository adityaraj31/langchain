from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump! The five boxing wizards jump quickly. Sphinx of black quartz, judge my vow.

This is a sample text with multiple paragraphs to demonstrate text splitting capabilities. It contains various sentences of different lengths and structures."""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
)

chunks = splitter.create_documents([text])

print(len(chunks))
print(chunks)