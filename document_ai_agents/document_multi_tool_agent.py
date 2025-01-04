from typing import Optional

import google.generativeai as genai
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from document_ai_agents.logger import logger


class ChromaEmbeddingsAdapter(Embeddings):
    def __init__(self, ef: EmbeddingFunction):
        self.ef = ef

    def embed_documents(self, texts):
        return self.ef(texts)

    def embed_query(self, query):
        return self.ef([query])[0]


class DocumentRAGState(BaseModel):
    question: str
    document_path: str
    pages_as_base64_jpeg_images: list[str]
    documents: list[Document]
    relevant_documents: list[Document] = Field(default_factory=list)
    response: Optional[str] = None


class DocumentRAGAgent:
    def __init__(self, model_name="gemini-1.5-flash-002", k=3):
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            self.model_name,
        )
        self.vector_store = Chroma(
            collection_name="document-rag",
            embedding_function=ChromaEmbeddingsAdapter(
                embedding_functions.DefaultEmbeddingFunction()
            ),
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

        self.graph = None
        self.build_agent()

    def index_documents(self, state: DocumentRAGState):
        assert state.documents, "Documents should have at least one element"

        if self.vector_store.get(where={"document_path": state.document_path})["ids"]:
            logger.info(
                "Documents for this file are already indexed, exiting this node"
            )

        self.vector_store.add_documents(state.documents)

    def answer_question(self, state: DocumentRAGState):
        relevant_documents: list[Document] = self.retriever.invoke(state.question)

        images = list(
            set(
                [
                    state.pages_as_base64_jpeg_images[doc.metadata["page_number"]]
                    for doc in relevant_documents
                ]
            )
        )  # Avoid duplicates

        logger.info(f"Responding to question {state.question}")
        messages = (
            [{"mime_type": "image/jpeg", "data": base64_jpeg} for base64_jpeg in images]
            + [doc.page_content for doc in relevant_documents]
            + [
                f"Answer this question using the context images and text elements only: {state.question}",
            ]
        )

        response = self.model.generate_content(messages)

        return {"response": response.text, "relevant_documents": relevant_documents}

    def build_agent(self):
        builder = StateGraph(DocumentRAGState)
        builder.add_node("index_documents", self.index_documents)
        builder.add_node("answer_question", self.answer_question)

        builder.add_edge(START, "index_documents")
        builder.add_edge("index_documents", "answer_question")
        builder.add_edge("answer_question", END)
        self.graph = builder.compile()


if __name__ == "__main__":
    from pathlib import Path

    from document_ai_agents.document_parsing_agent import (
        DocumentLayoutParsingState,
        DocumentParsingAgent,
    )

    state1 = DocumentLayoutParsingState(
        document_path=str(Path(__file__).parents[1] / "data" / "docs.pdf")
    )

    agent1 = DocumentParsingAgent()

    result1 = agent1.graph.invoke(state1)

    state2 = DocumentRAGState(
        question="Who was acknowledge in this paper ?",
        document_path=str(Path(__file__).parents[1] / "data" / "docs.pdf"),
        pages_as_base64_jpeg_images=result1["pages_as_base64_jpeg_images"],
        documents=result1["documents"],
    )

    agent2 = DocumentRAGAgent()

    result2 = agent2.graph.invoke(state2)

    print(result2["response"])

    state3 = DocumentRAGState(
        question="What is the macro average when fine tuning on publaynet using M-RCNN ? ",
        document_path=str(Path(__file__).parents[1] / "data" / "docs.pdf"),
        pages_as_base64_jpeg_images=result1["pages_as_base64_jpeg_images"],
        documents=result1["documents"],
    )

    result3 = agent2.graph.invoke(state3)

    print(result3["response"])
