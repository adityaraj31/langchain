from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

def main():
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        raise ValueError("Missing GROQ_API_KEY in .env file")

    model = ChatOpenAI(
        openai_api_key=groq_api_key,
        openai_api_base="https://api.groq.com/openai/v1",
        model_name="llama3-70b-8192"
    )

    chat_history = [
        SystemMessage(content="You are a helpful assistant."),

    ]

    print("Start chatting with the AI (type 'exit' to stop):")  
    while True:
        user_input = input('You: ')
        if user_input.lower() == 'exit':
            break

        chat_history.append(HumanMessage(content=user_input))
        response = model.invoke(chat_history)
        print("AI:", response.content)
        chat_history.append(AIMessage(content=response.content))

    print("\nChat history:")
    for msg in chat_history:
        role = msg.__class__.__name__.replace("Message", "")
        print(f"{role}: {msg.content}")


if __name__ == "__main__":
    main()
