from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community import chains
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY=os.getenv("GROQ_API_KEY")

def get_llm_chain(retriever):
    llm=ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="openai/gpt-oss-120b"
    )
    Prompt=PromptTemplate(
        input_variables=["context","question"],
        template="""
        🤖You are **MediBot** , an AI powered assistant trained to help users understand medical documents and
        health-related questions.
        your job is to provide clear , accurate and helpful response based **only on the provided context**.
        
        ----

        💭**context**:
        {context}

        🙋**User Question**:
        {question}
        ------------
        **Agent**:
        -respond in a calm ,factual, and respectful tone.
        -User simple explanation when needed
        -If the context does not contain the answer , say:"I'm sorry , but I couldn't find relevant information
        in the provided documents.
        -Do NOT make up the facts.
        -Do NOT give medical advice or diagnoses."
        """

    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt":Prompt},
        return_source_documents=True
    )