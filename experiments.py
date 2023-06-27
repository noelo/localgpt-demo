import os
import logging
import openai
from typing import List


from chromadb.utils import embedding_functions
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from constants import CHROMA_SETTINGS, SOURCE_DIRECTORY, PERSIST_DIRECTORY
from datetime import datetime
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

def load_single_document(file_path: str) -> List[Document]:
    # Loads a single document from a file path
    logging.info(f'Loading file {file_path}')
    if file_path.endswith(".adoc"):
        loader = TextLoader(file_path, encoding="utf8")
    elif file_path.endswith(".pdf"):
        loader = PyMuPDFLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
        
    return loader.load()


def load_documents(source_dir: str) -> List[Document]:
    result_docs=[]
    # Loads all documents from source documents directory
    all_files = os.listdir(source_dir)
    for file_path in all_files:
        result_docs.append(load_single_document(f"{source_dir}/{file_path}"))

    return result_docs
        
        
    
    # return [load_single_document(f"{source_dir}/{file_path}") for file_path in all_files if file_path[-4:] in ['.txt', '.pdf', '.csv','.adoc'] ]


def main():
        
    # openai.api_key = os.environ["OPENAI_API_KEY"]
    # model = INSTRUCTOR('hkunlp/instructor-large')
    embedding_function = HuggingFaceInstructEmbeddings()
    
    #uses base model and cpu
    # embedding_function = embedding_functions.InstructorEmbeddingFunction(model_name="hkunlp/instructor-large", device="cpu")

    # Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    flat_doc_list = sum(documents, [])
    logging.info(f'Num pages {len(flat_doc_list)}')
    # print(type(documents))
    # print(documents[0])
    # print(documents[1])
    
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # print("RecursiveCharacterTextSplitter Start :", datetime.now())
    # texts = text_splitter.split_documents(flat_doc_list)
    # print("RecursiveCharacterTextSplitter End :", datetime.now())
    # print(f"RecursiveCharacterTextSplitter Split into {len(texts)} chunks of text")
    
    text_splitter2 = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)
    print("TokenTextSplitter Start :", datetime.now())
    texts2 = text_splitter2.split_documents(flat_doc_list)
    print("TokenTextSplitter End :", datetime.now())
    print(f"Loaded {len(flat_doc_list)} documents from {SOURCE_DIRECTORY}")
    print(f"TokenTextSplitter Split into {len(texts2)} chunks of text")
    
    # embeddings = OpenAIEmbeddings()

    # # Create embeddings
    # # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
    # #                                             model_kwargs={"device": device})
    
    
    db = Chroma.from_documents(texts2, embedding_function, collection_name='OCP',persist_directory=PERSIST_DIRECTORY, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
