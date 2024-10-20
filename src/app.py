import os
import shutil
from dotenv import load_dotenv
from model.RAG.rag import RAG
from model.chat.chatbot import Bot

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model_options = {
    "1": "GPT 3.5-Turbo",
    "2": "GPT 4",
    "3": "Llama 3.1 by Groq",
    "4": "Llama-3.2-3B-Instruct",
    "5": "Llama-3.2-1B-Instruct"
}

model = input("Enter the model you want to use:\n1) GPT 3.5-Turbo\n2) GPT 4\n3) Llama 3.1 by Groq\n4) Llama-3.2-3B-Instruct\n5) Llama-3.2-1B-Instruct\n")

model = model_options.get(model)
api_key = ""
model_dir = ""

if model == "GPT 3.5-Turbo" or model == "GPT 4":
    api_key = OPENAI_API_KEY
elif model == "Llama 3.1 by Groq":
    api_key = GROQ_API_KEY
elif model == "Llama-3.2-3B-Instruct":
    api_key = GROQ_API_KEY
    model_dir = "E:\\WORKSPACE (PC)\\ia ii\\local_llm\\Llama-3.2-3B-Instruct"
elif model == "Llama-3.2-1B-Instruct":
    api_key = GROQ_API_KEY
    model_dir = "E:\\WORKSPACE (PC)\\ia ii\\local_llm\\Llama-3.2-1B-Instruct"


language = input("\nEnter the language you want to use (ES, EN): ")

rag = RAG()
chatbot = Bot(api_key, model, language, model_dir)


def chat():
    while True:
        message = input("\nEnter your message: ")

        if message == "exit":
            print("\nGoodbye :-)")
            break
        
        message = rag.augment_query(message)
        response = chatbot.chat(message)
        print(f"\nResponse: {response}")

chat()