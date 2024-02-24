from dotenv import dotenv_values 
from langchain_openai import OpenAIEmbeddings
from langchain_vectorstores.neo4j_vector import Neo4jVector
from langchain.schema import Document

# load values from .env
config = dotenv_values(".env")

# a list of documentos
documents = [
    Document(
        page_content="Text to indexed",
        metadata={"source": "local"}
    )
]

# service user to create the embeddings
embedding_provider = OpenAIEmbeddings(config["OPENIA_API_KEY"])

new_vector = Neo4jVector.from_documents(
    documents,
    embedding_provider,
    url=config["NEO4J_URI"],
    username=config["NEO4J_USERNAME"],
    password=config["NEO4J_PASSWORD"],
    index_name="myVectorIndex",
    node_label="Chunk",
    text_node_property="text",
    embedding_node_property="embedding",
    create_id_index=True,
)