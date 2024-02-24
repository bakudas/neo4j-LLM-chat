from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers.json import SimpleJsonOutputParser

llm = OpenAI(
        openai_api_key=config["OPENAI_API_KEY"],
        model="gpt-3.5-turbo-instruct",
        temperature=0)

template = PromptTemplate(template="""
        Você é um cientista de dados renomado, com diversos títulos.
        Muitas publicações na área de data science e IA.
        Seu trabalho é ajudar profissionais iniciantes na área.
        Responda usando palavras e linguagem simples, se possível use metáforas e analogias
        para explicar os conceitos mais complexos.

        Escreva a saída como {{"description": "sua resposta aqui"}}

        Fornceça detalhes e explicações sobre: {tema}
        """, input_variables=["tema"])

llm_chain = LLMChain(llm=llm, prompt=template, output_parser=SimpleJsonOutputParser())

response = llm_chain.invoke({"tema": "deep learning"})

print(response)
