from langchain_community.llms import Ollama
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA, LLMChain
from langchain_redis import RedisConfig, RedisVectorStore
from langchain.docstore.document import Document

import sys
import os
import time
import PyPDF2

TEMPLATE = """
            Use the following context and only the context to answer the query at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            Use one sentence maximum and keep the answer as concise as possible, but try to include the question context on the answer. 
            Dont leave the sentence unfinished, always finish the sentence.
            {context}
            Question: {query}
            Answer: 
        """

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

def separate_pdf_context(text):
    page_context = text[0].page_content
    
    # Split the text to get only the "Questions - context" part
    context_start = page_context.find("Questions - context")
    predefined_start = page_context.find("Questions Predefined")
    
    if context_start != -1 and predefined_start != -1:
        # Extract only the "Questions - context" section
        context_text = page_context[context_start:predefined_start]
        context_text = [Document(page_content=context_text)]
        
    else:
        print("Markers not found in the PDF file. Using full text as context.")
        context_text = text  # Fallback to full text if markers are not found
    
    return context_text

loader = PyPDFDirectoryLoader("butia_quiz\\resources\\2024")
docs = loader.load()
docs = separate_pdf_context(docs)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# Extract text content from each chunk
texts = [chunk.page_content for chunk in chunks]

embeddings = OllamaEmbeddings(model="nomic-embed-text")

config = RedisConfig(
    index_name="quiz-context",
    redis_url=REDIS_URL,
)

vector_store = RedisVectorStore(embeddings, config=config)

#ids = vector_store.add_texts(texts=texts)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 12})


ollama_configs = {
    'base_url': 'http://localhost:11434/',
    'model': "gemma2:2b-instruct-q5_1",
    'temperature': 0.4,
    'keep_alive': 600,
    
}

llm = Ollama(**ollama_configs)

prompt = ChatPromptTemplate.from_template(TEMPLATE)

rag_chain = (
    {"context": retriever,  "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

questions = ["When did the RoboCup@Home league start?",
"What are the main domains of focus in the RoboCup@Home league?",
"How is the performance of robots measured in the RoboCup@Home competition?",
"What is the population of Bahia?",
"Which areas in Bahia are characterized by the semi-arid climate?",
"What type of relief forms are predominant in Bahia?"]

print("-----------------------------------------")
for question in questions:
    print("Question: ", question)
    start = time.time()
    result = rag_chain.invoke(question)
    end = time.time()
    print("Time to answer: ", end - start)
    print("Answer: ",result)
    print("-----------------------------------------")