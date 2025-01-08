## Rag with PDF
# loads the document and split the text based on the lang chain function parameters 

# Import library
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassThrough 
from langchain_core.output_parsers import StrOutputParser 


loader = PyPDFLoader("rag_paper.pdf")
document = loader.load()

# Define a text splitter that splits recursively through the character list
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n', '.', ' ', ''],
    chunk_size=75,  
    chunk_overlap=10  
)

# Split the document using text_splitter
chunks = text_splitter.split_documents(document)
print(chunks)
print([len(chunk.page_content) for chunk in chunks])


# Initialize the OpenAI embedding model
embedding_model = OpenAIEmbeddings(api_key="api_key", model='text-embedding-3-small')

# Create a Chroma vector store and embed the chunks
vector_store = Chroma.from_documents(
    documents=chunks, 
    embedding=embedding_model
)

# Building a LCEL retrieval chain 
prompt = """
Use the only the context provided to answer the following question. If you don't know the answer, reply that you are unsure.
Context: {context}
Question: {question}
"""

# Convert the string into a chat prompt template
prompt_template = ChatPromptTemplate.from_template(prompt)

# Create an LCEL chain to test the prompt
chain = prompt_template | llm

# Invoke the chain on the inputs provided
print(chain.invoke({"context": "DataCamp's RAG course was created by Meri Nova and James Chapman!", "question": "Who created DataCamp's RAG course?"}))


# Convert the vector store into a retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Create the LCEL retrieval chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

# Invoke the chain
print(chain.invoke("Who are the authors?"))

## More sophisticated methods to build a 


# Create a document loader for README.md and load it
loader = UnstructuredMarkdownLoader("README.md")

markdown_data = loader.load()
print(markdown_data[0])


# Create a document loader for rag.py and load it
loader = PythonLoader('rag.py')

python_data = loader.load()
print(python_data[0])


# Create a Python-aware recursive character splitter
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=300, chunk_overlap=100
)

# Split the Python content into chunks
chunks = python_splitter.split_documents(python_data)

for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i+1}:\n{chunk.page_content}\n")


## Spliting on tokens 

import tiktoken 
from langchain_text_splitters import TokenTextSplitter 

example_string = "Mary had a little lamb, it's fleece was white as snow'

encoding = tiktoken.encoding_for_model('gpt-4o-mini')
splitter = TokenTextSplitter(encoding_name = enconding.name, 
                             chunk_size = 10, 
                             chunk_overlap = 2) 

chunks = splitter.split_text(example_string)

for i, chunk in enumerate(chunks): 
    print(f'Chunk {i+1}:\n[chunk]\n')

## Semanting splitting 

from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker 

embeddings = OpenAIEmbeddings(api_key="xxx", model = 'text-embedding-3-small')

semantinc_splitter = SemanticChunker(
    embeddings = embeddings, 
    breakpoint_threshold_type = 'gradient', 
    breakpoint_threshold_amount = 0.8)

chunks = semantic_splitter.split_documents(data)
print(chunks[0])

### Experiment with splitting methods 

# Get the encoding for gpt-4o-mini
encoding = tiktoken.encoding_for_model('gpt-4o-mini')

# Create a token text splitter
token_splitter = TokenTextSplitter(encoding_name=encoding.name, chunk_size=100, chunk_overlap=10)

# Split the PDF into chunks
chunks = token_splitter.split_documents(document)

for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i+1}:\nNo. tokens: {len(encoding.encode(chunk.page_content))}\n{chunk}\n")


# Instantiate an OpenAI embeddings model
embedding_model = OpenAIEmbeddings(api_key="<OPENAI_API_TOKEN>", model='text-embedding-3-small')

# Create the semantic text splitter with desired parameters
semantic_splitter = SemanticChunker(
    embeddings=embedding_model, breakpoint_threshold_type="gradient", breakpoint_threshold_amount=0.8
)

# Split the document
chunks = semantic_splitter.split_documents(document)
print(chunks[0])

## Sparse RAG retrieval (use with dictionaries) but low generalizable 

## BM25 retrieval 

from langchain_community.retrievers import BM25Retriever 

chunks = [
    "Python was created by Guido van Rossum and released in 1991.", 
    "Python is a popular language for machine learning", 
    "The PyTorch library is a popular Python library for AI and ML"
]

bm25_retriever = BM25Retriever.from_text(chunks, k = 3) 

results = bm25_retriever.invoke("When was Python created")

print(results[0].content()) 

## Retriever for documents and not chunks 


retriever = BM25Retriever.from_documents(
    documents = chunks, 
    k=5
)

chain = ({"context": retriever, "question": RunnablePassthrough()}
         | prompt
         | llm
         | StrOutoutParser() 
        )
## Example 

chunks = [
    "RAG stands for Retrieval Augmented Generation.",
    "Graph Retrieval Augmented Generation uses graphs to store and utilize relationships between documents in the retrieval process.",
    "There are different types of RAG architectures; for example, Graph RAG."
]

# Initialize the BM25 retriever
bm25_retriever = BM25Retriever.from_texts(chunks, k=3)

# Invoke the retriever
results = bm25_retriever.invoke("Graph RAG")

# Extract the page content from the first result
print("Most Relevant Document:")
print(results[0].page_content)


# Create a BM25 retriever from chunks

#Sparse retrieval with BM25
# Time to try out a sparse retrieval implementation! You'll create a BM25 retriever to ask questions about an academic paper on RAG, which has already been split into chunks called chunks. An OpenAI chat model and prompt have also been defined as llm and prompt, respectively. You can view the prompt provided by printing it in the console.


retriever = BM25Retriever.from_documents(
    documents=chunks, 
    k=5
)

# Create the LCEL retrieval chain
chain = ({"context": retriever, "question": RunnablePassthrough()}
         | prompt
         | llm
         | StrOutputParser()
)

# Invoke the chain
print(chain.invoke("What are knowledge-intensive tasks?"))








