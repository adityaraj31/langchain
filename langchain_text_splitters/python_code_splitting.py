from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """def calculate_sum(a, b):
    return a + b

def multiply_numbers(x, y):
    return x * y

class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, num):
        self.result += num
        return self.result
    
    def subtract(self, num):
        self.result -= num
        return self.result

# Test the calculator
calc = Calculator()
print(calc.add(5))
print(calc.subtract(2))

# Test basic functions
sum_result = calculate_sum(10, 20)
product = multiply_numbers(4, 5)
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0,
)

chunks = splitter.create_documents([text])

print(len(chunks))
print(chunks[0])