from dotenv import dotenv_values
# import langchain lib
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory

# load values from .env
config = dotenv_values(".env")

# setup LLM 
chat_llm = ChatOpenAI(
        openai_api_key=config["OPENAI_API_KEY"])

# create a prompt template 
prompt = PromptTemplate(template="""
        Você é um cientista de dados renomado, com diversos títulos.
        Muitas publicações na área de data science e IA.
        Seu trabalho é ajudar profissionais iniciantes na área.
        Responda usando palavras e linguagem simples, se possível use metáforas 
        e analogias para explicar os conceitos mais complexos.

        Histórico do Chat: {chat_history}
        Contexto: {context}
        Pergunta: {question}
        """, input_variables=["chat_history", "context", "question"])


# memory buffer to grant chat history
memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)

# create a chat chain to handle with prompts
chat_chain = LLMChain(llm=chat_llm, prompt=prompt, memory=memory, verbose=True)

# manage with context to perfom some grounding
# RAG for data science sandbox
data_science_context = """
        {
                "data science": [
                        {"skill": "Python", "description": "Linguagem de programação versátil com vasta biblioteca de análise de dados e machine learning, como Pandas, NumPy, Scikit-learn, TensorFlow e PyTorch."},
                        {"skill": "R", "description": "Linguagem de programação especializada em estatística e visualização de dados, amplamente usada em análises quantitativas e pesquisas acadêmicas."},
                        {"skill": "Data Mining", "description": "Processo de descoberta de padrões e conhecimentos a partir de grandes volumes de dados, utilizando técnicas como clustering, classificação e associação."},
                        {"skill": "Estatística e Probabilidade", "description": "Fundamentos matemáticos para entender a variabilidade dos dados, realizar testes hipotéticos e extrair inferências, essencial para análise de dados e modelagem preditiva."},
                        {"skill": "Visualização de Dados", "description": "Habilidade de representar dados de forma gráfica para facilitar a interpretação e comunicação de insights, utilizando ferramentas como Tableau, Power BI, Matplotlib e Seaborn."},
                        {"skill": "Manipulação de Dados", "description": "Técnicas de limpeza, transformação e preparação de dados para análise, incluindo tratamento de dados faltantes, normalização e codificação de variáveis."},
                        {"skill": "Banco de Dados", "description": "Conhecimento em sistemas de gerenciamento de banco de dados, SQL para consulta e manipulação de dados, e NoSQL para lidar com grandes volumes de dados não estruturados."}
                ],
                "IA": [
                        {"skill": "Machine Learning", "description": "Desenvolvimento de algoritmos que permitem que computadores aprendam a partir de dados, aplicando técnicas de regressão, classificação, clustering, entre outras."},
                        {"skill": "Deep Learning", "description": "Técnica avançada de machine learning que utiliza redes neurais profundas para modelar e entender complexos padrões de dados, amplamente usada em reconhecimento de imagem e linguagem natural."},
                        {"skill": "Processamento de Linguagem Natural (PLN)", "description": "Subcampo da IA focado na interação entre computadores e humanos através da linguagem natural, incluindo tradução automática, análise de sentimentos e chatbots."},
                        {"skill": "Visão Computacional", "description": "Uso de algoritmos para processar, analisar e compreender imagens do mundo real, aplicado em reconhecimento facial, veículos autônomos e diagnóstico médico por imagem."},
                        {"skill": "Reinforcement Learning", "description": "Área da IA que treina algoritmos usando um sistema de recompensas para tomar decisões sequenciais, útil em jogos, robótica e otimização de processos."},
                        {"skill": "Ética e IA Responsável", "description": "Compreensão dos impactos sociais da IA, incluindo viés algorítmico, privacidade de dados e uso ético da tecnologia, essencial para desenvolver soluções responsáveis e inclusivas."}
                ]
        }
        """

# simple loop for ask questions
while True:

        question = input("> ")

        # make request
        response = chat_chain.invoke({"context": data_science_context, "question": question})

        # print responses
        print(response["text"])
