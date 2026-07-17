import os
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_REGION = "us-east-1"
PINECONE_INDEX_NAME = "medicalindex"
INDEX_DIMENSION = 768

UPLOAD_DIR = Path("./uploaded_docs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Validate API keys
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing from .env")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from .env")


# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

spec = ServerlessSpec(
    cloud="aws",
    region=PINECONE_REGION,
)

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=INDEX_DIMENSION,
        metric="cosine",
        spec=spec,
    )

    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)


index = pc.Index(PINECONE_INDEX_NAME)


# Create one consistent embedding model
embed_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
    output_dimensionality=768,
)


def load_vectorstore(uploaded_files):
    saved_file_paths = []

    # 1. Save uploaded files
    for uploaded_file in uploaded_files:
        save_path = UPLOAD_DIR / uploaded_file.filename

        # Ensure pointer starts from beginning
        uploaded_file.file.seek(0)

        with open(save_path, "wb") as file_object:
            file_object.write(uploaded_file.file.read())

        saved_file_paths.append(save_path)

    # 2. Load, split, embed and upsert each PDF
    for pdf_path in saved_file_paths:
        print(f"Loading PDF: {pdf_path.name}")

        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            print(f"No content found in {pdf_path.name}")
            continue

        texts = [chunk.page_content for chunk in chunks]

        # Store text inside metadata so it can be retrieved later
        metadata = [
            {
                **chunk.metadata,
                "text": chunk.page_content,
                "source_file": pdf_path.name,
                "chunk_index": index_number,
            }
            for index_number, chunk in enumerate(chunks)
        ]

        # Correct IDs
        ids = [
            f"{pdf_path.stem}-{index_number}"
            for index_number in range(len(chunks))
        ]

        # 3. Generate embeddings
        print(f"Embedding {len(chunks)} chunks...")

        embeddings = embed_model.embed_documents(texts)

        dimensions = {len(vector) for vector in embeddings}

        print("Generated embedding dimensions:", dimensions)

        if dimensions != {INDEX_DIMENSION}:
            raise ValueError(
                f"Expected {INDEX_DIMENSION}-dimensional vectors, "
                f"but received: {dimensions}"
            )

        if not (
            len(ids)
            == len(embeddings)
            == len(metadata)
        ):
            raise ValueError(
                "IDs, embeddings and metadata counts do not match"
            )

        # 4. Prepare Pinecone vectors
        vectors = [
            {
                "id": vector_id,
                "values": vector_values,
                "metadata": vector_metadata,
            }
            for vector_id, vector_values, vector_metadata
            in zip(ids, embeddings, metadata)
        ]

        print(
            "Dimension being sent to Pinecone:",
            len(vectors[0]["values"]),
        )

        # Upsert in batches
        batch_size = 100

        with tqdm(
            total=len(vectors),
            desc=f"Upserting {pdf_path.name}",
        ) as progress:
            for start in range(0, len(vectors), batch_size):
                batch = vectors[start:start + batch_size]

                index.upsert(
                    vectors=batch,
                    namespace="medical-documents",
                )

                progress.update(len(batch))

        print(f"Upload complete for {pdf_path.name}")