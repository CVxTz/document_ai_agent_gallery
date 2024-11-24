from pathlib import Path

from PIL import Image

from document_ai_agents.nodes import extract_layout_elements, load_document
from document_ai_agents.states import DocumentLayoutParsingState, LayoutElement


def test_load_document_success():
    docs_path = Path(__file__).parents[2] / "data" / "docs.pdf"

    print(docs_path)
    state = DocumentLayoutParsingState(document_path=str(docs_path))
    result = load_document(state)
    assert "pages_as_images" in result
    assert "pages_as_texts" in result
    assert len(result["pages_as_images"]) > 0  # Expecting at least one page
    assert len(result["pages_as_texts"]) > 0


def test_extract_layout_elements_success():
    docs_path = Path(__file__).parents[2] / "data" / "docs.pdf"

    state = DocumentLayoutParsingState(document_path=str(docs_path))

    result_node1 = load_document(state)
    state.pages_as_images = result_node1["pages_as_images"]
    result_node2 = extract_layout_elements(state)

    images: list[Image] = result_node1["pages_as_images"]
    layout_elements: list[LayoutElement] = result_node2["layout_elements"]

    assert images
    assert layout_elements
