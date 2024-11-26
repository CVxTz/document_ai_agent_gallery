import concurrent.futures
import json
import operator
from pathlib import Path
from typing import Annotated, Literal

import google.generativeai as genai
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from document_ai_agents.document_utils import extract_images_from_pdf
from document_ai_agents.image_utils import pil_image_to_base64_jpeg
from document_ai_agents.logger import logger
from document_ai_agents.schema_utils import prepare_schema_for_gemini


class DetectedLayoutItem(BaseModel):
    element_type: Literal["Table", "Figure", "Image", "Text-block"] = Field(
        ...,
        description="Type of detected Item. Find Tables, figures and images. Use Text-Block for everything else, "
        "be as exhaustive as possible. Return 10 Items at most.",
    )
    summary: str = Field(..., description="A detailed description of the layout Item.")


class LayoutElements(BaseModel):
    layout_items: list[DetectedLayoutItem] = []


class DocumentLayoutParsingState(BaseModel):
    document_path: str
    pages_as_base64_jpeg_images: list[str] = []
    documents: Annotated[list[Document], operator.add] = []


class FindLayoutItemsInput(BaseModel):
    document_path: str
    base64_jpeg: str
    page_number: int


class DocumentParsingAgent:
    def __init__(self, model_name="gemini-1.5-flash-002"):
        layout_elements_schema = prepare_schema_for_gemini(LayoutElements)

        logger.info(f"Using Gemini model with schema: {layout_elements_schema}")
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            self.model_name,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": layout_elements_schema,
            },
        )
        self.graph = None
        self.threadpool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        self.build_agent()

    @classmethod
    def get_images(cls, state: DocumentLayoutParsingState):
        assert Path(state.document_path).is_file(), "File does not exist"

        images = extract_images_from_pdf(state.document_path)

        assert images, "No images extracted"

        pages_as_base64_jpeg_images = [pil_image_to_base64_jpeg(x) for x in images]

        return {"pages_as_base64_jpeg_images": pages_as_base64_jpeg_images}

    @classmethod
    def continue_to_find_layout_items(cls, state: DocumentLayoutParsingState):
        return [
            Send(
                "find_layout_items",
                FindLayoutItemsInput(
                    base64_jpeg=base64_jpeg,
                    page_number=i,
                    document_path=state.document_path,
                ),
            )
            for i, base64_jpeg in enumerate(state.pages_as_base64_jpeg_images)
        ]

    def find_layout_items(self, state: FindLayoutItemsInput):
        logger.info(f"Processing page {state.page_number + 1}")
        messages = [
            f"Find and summarize all the relevant layout elements in this pdf page in the following format: "
            f"{LayoutElements.model_json_schema()}. "
            f"Tables should have at least two columns and at least two rows. "
            f"The coordinates should overlap with each layout item.",
            {"mime_type": "image/jpeg", "data": state.base64_jpeg},
        ]

        result = self.model.generate_content(messages)
        data = json.loads(result.text)
        documents = [
            Document(
                page_content=x["summary"],
                metadata={
                    "page_number": state.page_number,
                    "element_type": x["element_type"],
                    "document_path": state.document_path,
                },
            )
            for x in data["layout_items"]
        ]

        logger.info(
            f"Extracted {len(data['layout_items'])} layout elements from page {state.page_number + 1}."
        )

        return {"documents": documents}

    def grouped_find_layout_items(self, state: DocumentLayoutParsingState):
        documents = []

        for i, base64_jpeg in enumerate(state.pages_as_base64_jpeg_images):
            result = self.find_layout_items(
                FindLayoutItemsInput(
                    document_path=state.document_path,
                    base64_jpeg=base64_jpeg,
                    page_number=i,
                )
            )
            documents.extend(result["documents"])

        return {"documents": documents}

    def build_agent(self):
        builder = StateGraph(DocumentLayoutParsingState)
        builder.add_node("get_images", self.get_images)
        builder.add_node("grouped_find_layout_items", self.grouped_find_layout_items)

        builder.add_edge(START, "get_images")
        builder.add_edge("get_images", "grouped_find_layout_items")
        builder.add_edge("grouped_find_layout_items", END)
        self.graph = builder.compile()


if __name__ == "__main__":
    _state = DocumentLayoutParsingState(
        document_path=str(Path(__file__).parents[1] / "data" / "docs.pdf")
    )

    agent = DocumentParsingAgent()

    result_node1 = agent.get_images(_state)
    _state.pages_as_base64_jpeg_images = result_node1["pages_as_base64_jpeg_images"]
    result_node2 = agent.find_layout_items(
        FindLayoutItemsInput(
            base64_jpeg=result_node1["pages_as_base64_jpeg_images"][0],
            page_number=0,
            document_path=str(_state.document_path),
        )
    )

    for item in result_node2["documents"]:
        print(item.page_content)
        print(item.metadata["element_type"])
