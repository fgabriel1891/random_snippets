## Rag with PDF
# loads the document and split the text based on the lang chain function parameters 

# Import library
from langchain_community.document_loaders import PyPDFLoader

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

