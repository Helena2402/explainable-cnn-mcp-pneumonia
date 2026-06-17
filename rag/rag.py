from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class ProjectRAG:
    """
    Retrieval-only component for project-specific documentation.
    Uses Chroma DB + sentence-transformer embeddings.
    """

    def __init__(self, db_path: str = "chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        self.db = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )

    def retrieve(self, query: str, k: int = 4) -> str:
        docs = self.db.similarity_search(query, k=k)

        if not docs:
            return ""

        return "\n\n".join([doc.page_content for doc in docs])
