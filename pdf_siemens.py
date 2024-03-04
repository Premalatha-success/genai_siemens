!pip3 install python-dotenv
!pip install langchain
!pip install unstructured
!pip install "unstructured[pdf]"
!pip install "unstructured[pdf]"
!pip install chromadb
!pip install tiktoken
!pip3 install openai
!pip install -U langchain-community
import openai

import os

deployment_name="gpt-35-turbo"
openai.api_key=""
openai.api_type="azure"
#openai.api_key="9c1e73115ede423ebf99c56d4e80a434"
#openai.api_base=""
openai.api_base=""
openai.api_version=""

from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI

os.environ["OPENAI_API_TYPE"]=openai.api_type
os.environ["OPENAI_API_Version"]=openai.api_version
os.environ["OPENAI_API_BASE"]=openai.api_base
os.environ["OPENAI_API_KEY"]=openai.api_key
load_dotenv

loader=UnstructuredFileLoader('C:/Users/Premalatha/Downloads/openai_new/Invoice12-edForce-Synechron_12days.pdf')
documents=loader.load()
documents

text_splitter=CharacterTextSplitter(chunk_size=900,chunk_overlap=0)
texts=text_splitter.split_documents(documents)
texts

embeddings=OpenAIEmbeddings()
doc_search=Chroma.from_documents(texts,embeddings)
chain=RetrievalQA.from_chain_type(llm=AzureOpenAI(model_kwargs={'engine':"gpt-35-turbo"}),chain_type="stuff",retriever=doc_search.as_retriever())


query="Item"
chain.run(query)