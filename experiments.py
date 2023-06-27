import os
import logging
import openai
from typing import List

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from constants import CHROMA_SETTINGS, SOURCE_DIRECTORY, PERSIST_DIRECTORY
from datetime import datetime


def load_documents(source_dir: str) -> List[Document]:
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    loader = PyPDFDirectoryLoader(path=source_dir,glob="*.pdf")
    docs = loader.load()
    logging.info(f"load_documents returning {len(docs)}")
    return docs

def main():
        
    openai.api_key = os.environ["OPENAI_API_KEY"]

    documents = load_documents(SOURCE_DIRECTORY)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    logging.info(f"RecursiveCharacterTextSplitter Start : {datetime.now()}")
    texts = text_splitter.split_documents(documents)
    logging.info("RecursiveCharacterTextSplitter End : {datetime.now()}")
    logging.info(f"RecursiveCharacterTextSplitter Split into {len(texts)} chunks of text")
       
    embedding = OpenAIEmbeddings()  
    
    db = Chroma.from_documents(texts, embedding,persist_directory=PERSIST_DIRECTORY, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
