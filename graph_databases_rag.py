from langchain_comunity.document_loaders import WikipediaLoader
from langchain_text_splitters import TokenTextSplitter 

raw_documents = WikipediaLoader(query="Large language model").load()
text_splitter = TokenTextSplitter(chunk_size = 100, chunk_overlap = 20)
documents = text_splitter.split_documents(raw_documents[:3])

print(documents[0])

# From text to graphs 

from langchain_openai import ChatOpenAI 
from langchain_experimental.graph_transformers import LLMGraphTransformer 

llm = ChatOpenAI(api_key="api", temperature=0, model_name="gpt-40-mini")
llm_transformer = LLMGraphTransformer(llm=llm)

graph_documents = llm_transformer.convert_to_graph_documents(documents)
print(graph_documents) 

## Case example:

# Define the LLM
llm = ChatOpenAI(api_key="<OPENAI_API_TOKEN>", model="gpt-4o-mini", temperature=0)

# Instantiate the LLM graph transformer
llm_transformer = LLMGraphTransformer(llm=llm)

# Convert the text documents to graph documents
graph_documents = llm_transformer.convert_to_graph_documents(docs)
print(f"Derived Nodes:\n{graph_documents[0].nodes}\n")
print(f"Derived Edges:\n{graph_documents[0].relationships}")

## Using Neo4j to store a graph node 

## Look into the set up of No4j 

from lagchain_community.graphs import Neo4jGraph 

graph = Neo4jGraph(url = "bolt://localhost:7687", username = "neo4j", password = 'xx' ) 

# Instantiate the Neo4j graph
graph = Neo4jGraph(url=url, username=user, password=password)

# Add the graph documents, sources, and include entity labels
graph.add_graph_documents(
  graph_documents, 
  include_source=True, 
  baseEntityLabel=True
)

graph.refresh_schema()

# Print the graph schema
print(graph.get_schema)

# Print the graph schema
print(graph.get_schema)

# Print the graph schema
print(graph.get_schema)

# Query the graph
results = graph.query("""
MATCH (relativity:Concept {id: "Theory Of Relativity"}) <-[:KNOWN_FOR]- (scientist)
RETURN scientist
""")

print(results[0])

## A graph RAG Application 

# Create the Graph Cypher QA chain
graph_qa_chain = GraphCypherQAChain.from_llm(
    llm=ChatOpenAI(api_key="<OPENAI_API_TOKEN>", temperature=0), graph=graph, verbose=True
)

# Invoke the chain with the input provided
result = graph_qa_chain.invoke({"query": "Who discovered the element Radium?"})

# Print the result text
print(f"Final answer: {result['result']}")



# Create the graph QA chain excluding Concept
graph_qa_chain = GraphCypherQAChain.from_llm(
    graph=graph, llm=llm, exclude_types=["Concept"], verbose=True
)

# Invoke the chain with the input provided
result = graph_qa_chain.invoke({"query": "Who was Marie Curie married to?"})
print(f"Final answer: {result['result']}")



# Create the graph QA chain, validating the generated Cypher query
graph_qa_chain = GraphCypherQAChain.from_llm(
    graph=graph, llm=llm, verbose=True, validate_cypher=True
)

# Invoke the chain with the input provided
result = graph_qa_chain.invoke({"query": "Who won the Nobel Prize In Physics?"})
print(f"Final answer: {result['result']}")

# Create an example prompt template
example_prompt = PromptTemplate.from_template(
    "User input: {question}\nCypher query: {query}"
)

# Create the few-shot prompt template
cypher_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.\n\nHere is the schema information\n{schema}.\n\nBelow are a number of examples of questions and their corresponding Cypher queries.",
    suffix="User input: {question}\nCypher query: ",
    input_variables=["question"]
)
