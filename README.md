# Langchain RAG with Chromadb

## Author

Yuluka Gigante Muriel


## Overview

This repository contains a simple RAG implementation, that allows you to interact with a specialized chatbot, via the CLI.

You'll find that there are five available models to interact with. Three of those models are hosted by a third party company, while the last two are meant to be self hosted. 


## How to use it

To use this project you must follow these steps:

1. Install the dependencies listed in 'requirements.txt':

    ```bash
    pip install -r requirements.txt
    ```

    > **Note**: This project is configured for a system with an NVIDIA GPU using PyTorch (CUDA 11.8) on Windows OS. If you're on a different platform, adjust the installation accordingly by referring to the [PyTorch website](https://pytorch.org).

2. Download the models:

    If you want to use the RAG with a local LLM, you'll need to download the model.

    [Here](https://github.com/yuluka/local_llm_chatbot) you can how to setup that.

3. Get the necessary API keys:

    As the other models are not self-hosted, it is necessary to get API keys to use them. In this case, I'm using models from Groq and OpenAI, so you can get the keys on:

    - [Groq](https://console.groq.com/keys)
    - [OpenAI](https://platform.openai.com/api-keys)

    Once you have the key you'll use, create a `.env` file at the root of the project, and write `OPENAI_API_KEY=«the_openai_key»` or `GROQ_API_KEY=«the_groq_key»`.

    **Note:** To use the OpenAI models, it is necessary to pay (minimum $5).

4. Collect and store the data:

    You'll need to collect the information you want your model to specialize in. It can be any text information, unless you use a vision model.

    The data must be placed in a folder called `data`, at the root of the project. I'm using markdown documents, so you can put your data in that form, or change the way the documents are loaded:

    ```python
    def load_documents(self) -> list[Document]:
        """	
        Load the documents from the data directory.

        :return: The documents.
        :rtype: list[Document]
        """
        
        loader = DirectoryLoader(self.DATA_PATH, glob="*.md") # Change glob value to accept other type of docs
        documents = loader.load()
        
        return documents
    ```

5. Run the `app.py` script:

    To start the chatbot, run the following command:

    ```bash
    python ./src/app.py
    ```

    Once the script is running, you'll be prompted to choose a model to interact with. The first time you run it, it'll create the embeddings database, so you'll have to wait a moment.


I hope you find this useful.