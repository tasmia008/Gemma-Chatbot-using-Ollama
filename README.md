# Gemma-Chatbot-using-Ollama

This repository provides an implementation of a chatbot named "Gemma" using the Ollama model. The chatbot is designed to load documents from specified URLs, split the text into manageable chunks, convert them to embeddings, and retrieve relevant documents for answering user queries.

## Installation

Ensure you have Python 3.8 or above installed. You can install the required packages using pip:

```bash
pip install langchain_community chroma
Code Overview
##The main components of the code are as follows:

##Loading Data from URLs: The script loads web pages from a list of URLs and extracts text from them.
##Splitting Data into Chunks: The extracted text is split into smaller chunks to make it manageable for processing.
##Converting Documents to Embeddings: The text chunks are converted into embeddings and stored in a vector store for efficient retrieval.
##Retrieving Relevant Documents: The vector store is used to retrieve documents relevant to the user's query.
##Question Answering: The chatbot answers questions using the retrieved documents as context.
##Step-by-Step Guide
##1. Load Data from URLs
The URLs from which the data is to be extracted are specified in a list. The WebBaseLoader class is used to load the data from these URLs.


from langchain_community.document_loaders import WebBaseLoader

urls = [
    "https://ollama.com/",
    "https://ollama.com/blog/windows-preview",
    "https://ollama.com/blog/opensi-compatibility"
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
##2. Split Data into Chunks
The CharacterTextSplitter class is used to split the text into chunks of 7500 characters with an overlap of 100 characters.

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs_list)
##3. Convert Documents to Embeddings and Store Them
The text chunks are converted to embeddings using the Ollama model and stored in a Chroma vector store.

from langchain_community.vectorstores import chroma
from langchain_community import embeddings

vectorstore = chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embeddings=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text')
)
retriever = vectorstore.as_retriever()
##4. Question Answering Before and After Retrieval-Augmented Generation (RAG)
##Before RAG:

A basic question answering setup using the Ollama model without any additional context.


from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

model_local = ChatOllama(model="gemma")

before_rag_template = "what is {topic}"
before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)
before_rag_chain = before_rag_prompt | model_local | StrOutputParser()

print(before_rag_chain.invoke({"topic": "Ollama"}))
##After RAG:

A more advanced setup where the chatbot answers questions using the context retrieved from the vector store.


from langchain_core.runnables import RunnablePassthrough

after_rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""

after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt
    | model_local
    | StrOutputParser()
)

print(after_rag_chain.invoke("What is Ollama"))
Conclusion
This implementation demonstrates how to create a chatbot using the Ollama model with retrieval-augmented generation (RAG) to enhance its responses. The chatbot can load data from web pages, process it into manageable chunks, convert it to embeddings, store it in a vector store, and use it to answer user queries effectively.
