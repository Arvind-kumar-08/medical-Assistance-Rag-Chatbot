import os
from typing import Any, List

from dotenv import load_dotenv
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone

from logger import logger
from modules.llm import get_llm_chain
from modules.query_handler import query_chain


load_dotenv()

router = APIRouter()


# Must match the dimension of your Pinecone index
EMBEDDING_DIMENSION = 768

# Must match the namespace used while uploading PDF vectors
PINECONE_NAMESPACE = os.getenv(
    "PINECONE_NAMESPACE",
    "medical-documents",
)


class SimpleRetriever(BaseRetriever):
    """
    Simple LangChain retriever which returns the documents
    already fetched from Pinecone.
    """

    documents: List[Document]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> List[Document]:
        return self.documents


@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        question = question.strip()

        if not question:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Question cannot be empty",
                },
            )

        logger.info(f"User query: {question}")

        # Read environment variables
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
        google_api_key = os.getenv("GOOGLE_API_KEY")

        if not pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY is missing from the .env file"
            )

        if not pinecone_index_name:
            raise ValueError(
                "PINECONE_INDEX_NAME is missing from the .env file"
            )

        if not google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing from the .env file"
            )

        # Pinecone setup
        pc = Pinecone(
            api_key=pinecone_api_key,
        )

        index = pc.Index(
            pinecone_index_name,
        )

        # Log Pinecone index information for debugging
        index_stats = index.describe_index_stats()

        logger.info(
            f"Pinecone index stats: {index_stats}"
        )

        logger.info(
            f"Searching namespace: {PINECONE_NAMESPACE}"
        )

        # Gemini embedding model
        # Must use the same model and dimension as PDF upload
        embed_model = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=google_api_key,
            output_dimensionality=EMBEDDING_DIMENSION,
        )

        # Generate query embedding
        embedded_query = embed_model.embed_query(
            question
        )

        logger.info(
            f"Query embedding dimension: "
            f"{len(embedded_query)}"
        )

        # Prevent Pinecone dimension mismatch
        if len(embedded_query) != EMBEDDING_DIMENSION:
            raise ValueError(
                "Embedding dimension mismatch. "
                f"Expected {EMBEDDING_DIMENSION}, "
                f"received {len(embedded_query)}"
            )

        # Query Pinecone
        response = index.query(
            vector=embedded_query,
            top_k=3,
            include_metadata=True,
            include_values=False,
            namespace=PINECONE_NAMESPACE,
        )

        # Pinecone can return an object response
        matches = getattr(
            response,
            "matches",
            None,
        )

        # Dictionary response fallback
        if matches is None and isinstance(response, dict):
            matches = response.get(
                "matches",
                [],
            )

        matches = matches or []

        logger.info(
            f"Pinecone matches found: {len(matches)}"
        )

        if not matches:
            return JSONResponse(
                status_code=404,
                content={
                    "error": (
                        "No relevant information was found. "
                        f"Check that vectors exist inside the "
                        f"'{PINECONE_NAMESPACE}' namespace."
                    ),
                },
            )

        documents: List[Document] = []

        # Convert Pinecone matches into LangChain documents
        for match in matches:
            metadata = getattr(
                match,
                "metadata",
                None,
            )

            # Dictionary response fallback
            if metadata is None and isinstance(match, dict):
                metadata = match.get(
                    "metadata",
                    {},
                )

            metadata = metadata or {}

            # Text was stored in metadata during PDF upload
            page_content = metadata.get(
                "text",
                "",
            )

            if not isinstance(page_content, str):
                page_content = str(page_content)

            page_content = page_content.strip()

            if not page_content:
                logger.warning(
                    "A Pinecone match was found, but its "
                    "metadata does not contain the 'text' field"
                )
                continue

            score = getattr(
                match,
                "score",
                None,
            )

            if score is None and isinstance(match, dict):
                score = match.get("score")

            document_metadata = {
                **metadata,
                "pinecone_score": score,
            }

            documents.append(
                Document(
                    page_content=page_content,
                    metadata=document_metadata,
                )
            )

        logger.info(
            f"LangChain documents created: {len(documents)}"
        )

        if not documents:
            return JSONResponse(
                status_code=404,
                content={
                    "error": (
                        "Vectors were found, but their metadata "
                        "does not contain document text. "
                        "Re-upload the PDF after adding the "
                        "'text' field to Pinecone metadata."
                    ),
                },
            )

        # Create custom retriever
        retriever = SimpleRetriever(
            documents=documents,
        )

        # Create RAG chain
        chain = get_llm_chain(
            retriever,
        )

        # Generate final response
        result = query_chain(
            chain,
            question,
        )

        logger.info(
            "Query processed successfully"
        )

        return result

    except Exception as error:
        logger.exception(
            "Error processing question"
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": str(error),
            },
        )