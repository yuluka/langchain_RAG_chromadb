from groq import Groq
from openai import OpenAI
from transformers import pipeline
import os
import torch

class Bot:
    """
    Class Bot is a class that represents a chatbot. It is responsible for handling the conversation with the user and sending messages to the model.

    The Bot class is used to interact with the OpenAI or Llama3 models.
    """

    def __init__(self, api_key: str, model: str, language: str, model_dir: str = None):
        self.API_KEY = api_key
        self.MODEL = model
        self.LANGUAGE = language
        self.MODEL_DIR = model_dir

        self.init_config()

    def init_config(self):
        """
        Initializes the configuration of the bot.

        Stablishes the context of the conversation and initializes the OpenAI or Llama3 client, depending on the model selected by the user.
        """

        if self.LANGUAGE == "EN":
            context = "You are a virtual assistant. You must speak english. Your manner must be cordial."
        elif self.LANGUAGE == "ES":
            context = "Eres un asistente virtual. Debes hablar en español. Tu trato debe ser cordial."

        self.message_history = [{"role": "system", "content": context}]

        if self.MODEL == "GPT 3.5-Turbo" or self.MODEL == "GPT 4":
            self.init_openai()
        elif self.MODEL == "Llama 3.1 by Groq":
            self.init_groq_llama3()
        else:
            self.init_pipeline()

    def init_openai(self):
        """
        Initializes the OpenAI client.
        """

        self.client = OpenAI(
            api_key=self.API_KEY,
        )

    def init_groq_llama3(self):
        """
        Initializes the Llama3 client.
        """

        self.client = Groq(
            api_key=self.API_KEY,
        )

    def init_pipeline(self):
        """
        Initializes the pipeline for the model.
        """

        self.pipe = pipeline(
            'text-generation',
            model=self.MODEL_DIR,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
            },
            device="cuda",
        )


    def chat(self, message: str) -> str:
        """
        Sends a message to the model and returns the response.

        :param message: The message to send to the model.
        :type message: str
        :return: The response from the model.
        :rtype: str
        """

        if self.MODEL == "GPT 3.5-Turbo" or self.MODEL == "GPT 4":
            return self.chat_openai(message)
        elif(self.MODEL == "Llama 3.1 by Groq"):
            return self.chat_llama3(message)
        else:
            return self.chat_local(message)

    def chat_openai(self, message: str) -> str:
        """
        Sends a message to the OpenAI model and returns the response.

        :param message: The message to send to the model.
        :type message: str
        :return: The response from the model.
        :rtype: str
        """

        pass

    def chat_llama3(self, message: str) -> str:
        """
        Sends a message to the llama3 model and returns the response.

        :param message: The message to send to the model.
        :type message: str
        :return: The response from the model.
        :rtype: str
        """

        self.message_history.append({"role": "user", "content": message})

        completion = self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=self.message_history,
            temperature=1,
            max_tokens=8192,
            top_p=1,
            stream=True,
            stop=None,
        )

        response = ""

        for chunk in completion:
            response += chunk.choices[0].delta.content or ""

        self.message_history.append({"role": "assistant", "content": response})

        return response


    def chat_local(self, message: str) -> str:
        """
        Sends a message to the local model and returns the response.

        :param message: The message to send to the model.
        :type message: str
        :return: The response from the model.
        :rtype: str
        """

        self.message_history.append({"role": "user", "content": message})

        response = self.pipe(
            self.message_history,
            max_new_tokens=8192,
            do_sample=False,
        )

        response = response[0]["generated_text"][-1]["content"]

        self.message_history.append({"role": "assistant", "content": response})

        return response