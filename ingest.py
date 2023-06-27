import os
from typing import List
import openai


from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from constants import CHROMA_SETTINGS, SOURCE_DIRECTORY, PERSIST_DIRECTORY
from datetime import datetime

def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    if file_path.endswith(".adoc"):
        loader = TextLoader(file_path, encoding="utf8")
    elif file_path.endswith(".pdf"):
        loader = PDFMinerLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    return loader.load()[0]


def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    all_files = os.listdir(source_dir)
    return [load_single_document(f"{source_dir}/{file_path}") for file_path in all_files if file_path[-4:] in ['.txt', '.pdf', '.csv','.adoc'] ]


def main():
        
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # Load documents and split in chunks
    print(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_splitter2 = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)
    print("RecursiveCharacterTextSplitter Start :", datetime.now())
    texts = text_splitter.split_documents(documents)
    print("RecursiveCharacterTextSplitter End :", datetime.now())
    print("TokenTextSplitter Start :", datetime.now())
    texts2 = text_splitter2.split_documents(documents)
    print("TokenTextSplitter End :", datetime.now())
    print(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    print(f"RecursiveCharacterTextSplitter Split into {len(texts)} chunks of text")
    print(f"TokenTextSplitter Split into {len(texts2)} chunks of text")
    
    embeddings = OpenAIEmbeddings()

    # # Create embeddings
    # # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
    # #                                             model_kwargs={"device": device})
    
    db = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None


if __name__ == "__main__":
    main()
