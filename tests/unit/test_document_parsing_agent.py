from pathlib import Path

from document_ai_agents.document_parsing_agent import (
    DocumentLayoutParsingState,
    DocumentParsingAgent,
)


def test_load_document_success():
    docs_path = Path(__file__).parents[2] / "data" / "docs.pdf"

    state = DocumentLayoutParsingState(document_path=str(docs_path))
    agent = DocumentParsingAgent()
    result = agent.get_images(state)
    assert "pages_as_base64_png_images" in result
    assert len(result["pages_as_base64_png_images"]) > 0  # Expecting at least one page


def test_extract_layout_elements_success():
    docs_path = Path(__file__).parents[2] / "data" / "docs.pdf"

    state = DocumentLayoutParsingState(document_path=str(docs_path))
    agent = DocumentParsingAgent()
    result_images = agent.get_images(state)
    state.pages_as_base64_png_images = result_images["pages_as_base64_png_images"]
    result = agent.find_layout_items(state)
    assert len(result["documents"]) > 0  # Expecting at least one item
