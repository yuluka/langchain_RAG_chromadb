import os

import unstructured
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class RAG:
    """
    Retrieval Augmented Generation (RAG) model.

    This class is used to generate embeddings for a set of documents, retrieve relevant information based on a query, and augment the query with the 
    relevant information.
    """

    def __init__(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.current_dir = os.path.dirname(self.current_dir)
        self.current_dir = os.path.dirname(self.current_dir)
        self.current_dir = os.path.dirname(self.current_dir)

        self.DATA_PATH = os.path.join(self.current_dir, "data")
        self.CHROMA_PATH = os.path.join(self.current_dir, "chroma")
        self.PROMPT_TEMPLATE = """
Answer the question based only on the context below:

{context}

-----------------------------------

Answer the question based on the above context: {question}
"""

        if not os.path.exists(self.CHROMA_PATH):
            documents = self.load_documents()
            chunks = self.split_documents(documents)
            self.db = self.generate_n_save_embeddings(chunks)
        else:
            self.db = self.generate_n_save_embeddings([])

    def load_documents(self) -> list[Document]:
        """	
        Load the documents from the data directory.

        :return: The documents.
        :rtype: list[Document]
        """
        
        loader = DirectoryLoader(self.DATA_PATH, glob="*.md")
        documents = loader.load()
        
        return documents

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split the documents into chunks.

        :param documents: The documents to split.
        :type documents: list[Document]
        :return: The split documents.
        :rtype: list[Document]
        """

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )

        chunks = text_splitter.split_documents(documents)

        return chunks

    def generate_n_save_embeddings(self, chunks: list[Document]) -> Chroma:
        """
        Generate and save the embeddings for the documents in a Chroma database.

        :param chunks: The documents to generate embeddings for.
        :type chunks: list[Document]
        :return: The Chroma database.
        :rtype: Chroma
        """

        if os.path.exists(self.CHROMA_PATH):
            db = Chroma(
                persist_directory=self.CHROMA_PATH,
                embedding_function=OpenAIEmbeddings(),
            )

            return db

        db = Chroma.from_documents(
            documents=chunks, 
            embedding=OpenAIEmbeddings(),
            persist_directory=self.CHROMA_PATH,
        )
        
        return db

    def retrieve_relevant_info(self, db: Chroma, query: str, n: int = 3) -> list[Document]:
        """
        Retrieve relevant information based on a query.

        :param db: The Chroma database.
        :type db: Chroma
        :param query: The query to search for.
        :type query: str
        :param n: The number of results to return. Defaults to 3.
        :type n: int
        :return: The relevant information.
        :rtype: list[Document]
        """

        results = db.similarity_search(
            query=query,
            k=n,
        )

        if len(results) == 0:
            return []

        return results

    def augment_query(self, query: str) -> str:
        """
        Augment the query with relevant information.

        :param query: The query to augment.
        :type query: str
        :return: The augmented query.
        :rtype: str
        """

        relevant_info = self.retrieve_relevant_info(self.db, query)

        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_info])
        prompt_template = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)

        augmented_query = prompt_template.format(
            context=context,
            question=query,
        )

        return augmented_query
    
    def get_db(self):
        """
        Get the Chroma database.

        :return: The Chroma database.
        :rtype: Chroma
        """

        return self.db