from pathlib import Path

from document_ai_agents.document_parsing_agent import (
    DocumentLayoutParsingState,
    DocumentParsingAgent,
    FindLayoutItemsInput,
)


def test_load_document_success():
    docs_path = Path(__file__).parents[2] / "data" / "docs.pdf"

    state = DocumentLayoutParsingState(document_path=str(docs_path))
    agent = DocumentParsingAgent()
    result = agent.get_images(state)
    assert "pages_as_base64_jpeg_images" in result
    assert len(result["pages_as_base64_jpeg_images"]) > 0  # Expecting at least one page


def test_extract_layout_elements_success():
    docs_path = Path(__file__).parents[2] / "data" / "docs.pdf"

    state = DocumentLayoutParsingState(document_path=str(docs_path))
    agent = DocumentParsingAgent()
    result_images = agent.get_images(state)
    state.pages_as_base64_jpeg_images = result_images["pages_as_base64_jpeg_images"]
    result = agent.find_layout_items(
        FindLayoutItemsInput(
            base64_jpeg=result_images["pages_as_base64_jpeg_images"][0],
            page_number=0,
            document_path=state.document_path,
        )
    )
    assert len(result["documents"]) > 0  # Expecting at least one item


def test_document_parser_agent():
    docs_path = Path(__file__).parents[2] / "data" / "docs.pdf"

    state = DocumentLayoutParsingState(document_path=str(docs_path))
    agent = DocumentParsingAgent()

    result = agent.graph.invoke(state)

    assert len(result["documents"]) > 0  # Expecting at least one item
