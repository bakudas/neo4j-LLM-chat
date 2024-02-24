from dotenv import dotenv_values
# import langchain lib
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_community.tools import YouTubeSearchTool
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector

# load values from .env
config = dotenv_values(".env")

# setup LLM 
chat_llm = ChatOpenAI(openai_api_key=config["OPENAI_API_KEY"])

# setup embedding provider
embedding_provider = OpenAIEmbeddings(openai_api_key=config["OPENAI_API_KEY"])

movie_plot_vector = Neo4jVector.from_existing_index(
        embedding_provider,
        url=config["NEO4J_URI"],
        username=config["NEO4J_USERNAME"],
        password=config["NEO4J_PASSWORD"],
        index_name="moviePlots",
        embedding_node_property="embedding",
        text_node_property="plot",
)

plot_retriever = RetrievalQA.from_llm(
    llm=chat_llm,
    retriever=movie_plot_vector.as_retriever(),
    verbose=True,
    return_source_documents=True
)

def run_retriever(query):
    results = plot_retriever.invoke({"query":query})
    # format the results
    movies = '\n'.join([doc.metadata["title"] + " - " + doc.page_content for doc in results["source_documents"]])
    return movies       

# create a prompt template 
prompt = PromptTemplate(template="""
        You are a movie expert. You find movies from a genre or plot.

        Histórico do Chat: {chat_history}
        Pergunta: {input}
        """, input_variables=["chat_history", "input"])


# memory buffer to grant chat history
memory = ConversationBufferMemory(memory_key="chat_history", input_key="input", return_messages=True)

# create a chat chain to handle with prompts
chat_chain = LLMChain(llm=chat_llm, prompt=prompt, memory=memory, verbose=True)

# initialising specific tools
youtube = YouTubeSearchTool()

# manage tools the model will able to use
tools = [
    Tool.from_function(
        name="Movie Chat",
        description="For when you need to chat about movies. The question will be a string. Return a string.",
        func=chat_chain.run,
        return_direct=True
    ),
    Tool.from_function(
        name="Movie Trailer Search",
        description="Use when needing to find a movie trailer. The question will include the word 'trailer'. Return a link to a YouTube video.",
        func=youtube.run,
        return_direct=True
    ),
    Tool.from_function(
        name="Movie Plot Search",
        description="For when you need to compare a plot to a movie. The question will be a string. Return a string.",
        func=run_retriever,
        return_direct=True
    )
]

# create and initialise agents
agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(chat_llm, tools, agent_prompt)
agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        memory=memory,
        max_interations=3,
        verbose=True,
        handle_parse_errors=True,
        )

# simple loop for ask questions
while True:

        question = input("> ")

        # make request
        response = agent_executor.invoke({"input": question})

        # print responses
        print(response["output"])
