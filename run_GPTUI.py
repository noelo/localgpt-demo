import os
import openai
import chainlit as cl
from chainlit import AskUserMessage, Message, on_chat_start
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY



SYSTEM_TEMPLATE = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

```
The answer is foo
SOURCES: xyz
```

Begin!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


@on_chat_start
async def main():
        Message(
            content=f"Ask questions to the OpenShift Documentation",
        ).send()


@cl.langchain_factory(use_async=True)
def load_model():
    openai.api_key = os.environ["OPENAI_API_KEY"]
    local_llm = OpenAI(temperature=0.0)
    embeddings = OpenAIEmbeddings()
    x = Chroma.as_retriever
    # load the vectorstore
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=local_llm, collection_name='OCP',chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa


@cl.langchain_postprocess
def process_response(res):
    answer = res["result"]
    sources = res["source_documents"]
    source_elements = []
    found_sources = []

    # Get the metadata and texts from the user session
    # metadatas = cl.user_session.get("metadatas")
    # all_sources = [m["source"] for m in metadatas]
    # texts = cl.user_session.get("texts")

    if sources:


    #     # Add the sources to the message
        i = 0
        for source in sources:
            # print(source)
            
    #         # Get the index of the source
    #         try:
    #             index = all_sources.index(source_name)
    #         except ValueError:
    #             continue
    #         text = texts[index]
            found_sources.append(source.metadata)
    #         # Create the text element referenced in the message
            source_elements.append(cl.Text(id=i,text=source.metadata['source'], name=source.metadata['source'], display="side"))
            i+=1

        # if found_sources:
        #     print(found_sources)
        #     answer += f"\nSources: {', '.join(found_sources)}"
        # else:
        #     answer += "\nNo sources found"
        
    x = []
    text_content = "Hello, this is a text element."
    
    for src in found_sources:
        print(src)
        src_str = src['source']
        res_str = src_str.replace("/home/noelo/dev/localGPT/SOURCE_DOCUMENTS/", "")

        x.append(cl.Text(name=res_str, text="https://docs.openshift.com", display="inline"))

    print(source_elements)
    cl.Message(content=answer, elements=x).send()

    # cl.Text(name="simple_text", text=text_content, display="inline").send()
    # cl.Text(name="simple_text", text="this is a test", display="inline").send()