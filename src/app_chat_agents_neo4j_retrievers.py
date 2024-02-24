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
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector

# load values from .env
config = dotenv_values(".env")

# # graph db setup
# graph = Neo4jGraph(
#         url=config["NEO4J_URI"],
#         username=config["NEO4J_USERNAME"],
#         password=config["NEO4J_PASSWORD"],
# )

# # print graph schema 
# print(graph.schema)

# result = graph.query("""
# MATCH (m:Movie{title: 'Toy Story'}) 
# RETURN m.title, m.plot, m.poster
# """)

# # print the cypher query result
# print(result)

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

result = plot_retriever.invoke(
    {"query": "você poderia me sugerir um filme sobre guerra intergalática?"}
)

print(result)

# create a prompt template 
prompt = PromptTemplate(template="""
        Suas áreas de interesse são: Design, Research, Science, Data Science, Tecnology, Computação de Alto Desempenho.
        Você é um cientista de dados renomado, com diversos títulos.
        Muitas publicações na área de data science e IA.
        Seu trabalho é ajudar profissionais iniciantes na área.
        Responda usando palavras e linguagem simples, se possível use metáforas 
        e analogias para explicar os conceitos mais complexos.
        Você sempre responderá em pt-br.

        Histórico do Chat: {chat_history}
        Pergunta: {input}
        """, input_variables=["chat_history", "input"])


# memory buffer to grant chat history
memory = ConversationBufferMemory(memory_key="chat_history", input_key="input", return_messages=True)

# create a chat chain to handle with prompts
chat_chain = LLMChain(llm=chat_llm, prompt=prompt, memory=memory, verbose=True)

# initialising specific tools
youtube = YouTubeSearchTool()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# manage tools the model will able to use
tools = [
        Tool.from_function(
                name="Chat Normal sobre Data Science, Desing, Programação e IA",
                description="Use sempre para conversas sobre a área de data science e IA, design, programação e conhecimentos gerais. A pergunta será uma string. Retorne uma string.",
                func=chat_chain.run,
                return_direct=True,
        ),
        Tool.from_function(
                name="Buscador de Tutorial e Vídeos instrutivos",
                description="Use apenas quando for preciso buscar um tutorial ou vídeo. A pergunta precisa incluir a palavra 'tutorial' e/ou 'curso' e/ou 'video'. Retorne com uma breve explicação e um link para para um vídeo no Youtube.",
                func=youtube.run,
                return_direct=True,
        ),
        Tool.from_function(
                name="Buscador de artigos na Wikepedia.",
                description="Use apenas quando for estritamente necessário buscar uma referência na wikipedia. A pergunta precisa incluir a palavra 'wikipedia'. Retorne com as informações da Wikepedia.",
                func=wikipedia.run,
                return_direct=True,
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
