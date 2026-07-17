from logger import logger


def query_chain(chain, question: str) -> dict:
    """
    Execute the RAG chain and return a JSON-compatible response.
    """

    result = chain.invoke(
        {
            "query": question,
        }
    )

    logger.info(f"Raw chain result: {result}")

    answer = result.get("result", "")

    if not isinstance(answer, str):
        answer = str(answer)

    source_documents = result.get(
        "source_documents",
        [],
    )

    sources = []

    for document in source_documents:
        metadata = getattr(
            document,
            "metadata",
            {},
        ) or {}

        source = metadata.get(
            "source_file",
            metadata.get(
                "source",
                "Unknown source",
            ),
        )

        page = metadata.get("page")
        score = metadata.get("pinecone_score")

        source_data = {
            "source": str(source),
        }

        if page is not None:
            source_data["page"] = page

        if score is not None:
            source_data["score"] = float(score)

        sources.append(source_data)

    # Remove duplicate sources
    unique_sources = []
    seen_sources = set()

    for source_data in sources:
        source_key = (
            source_data.get("source"),
            source_data.get("page"),
        )

        if source_key not in seen_sources:
            seen_sources.add(source_key)
            unique_sources.append(source_data)

    response = {
        "success": True,
        "question": question,
        "answer": answer,
        "sources": unique_sources,
    }

    logger.info(f"Final API response: {response}")

    return response