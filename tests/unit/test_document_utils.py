import pytest
from pathlib import Path
from document_ai_agents.document_utils import extract_images_from_pdf, extract_text_from_pdf



@pytest.fixture
def pdf_file():
    docs_path = Path(__file__).parents[2] / "data" / "docs.pdf"
    assert docs_path.exists(), f"PDF file not found at {docs_path}"
    return str(docs_path)

# Test for extract_images_from_pdf
def test_extract_images_from_pdf(pdf_file):
    images = extract_images_from_pdf(pdf_file)
    assert len(images) > 0, "Expected at least one image in the PDF"
    assert all(image.format == "PNG" for image in images), "Images should be in PNG format"

# Test for extract_text_from_pdf
def test_extract_text_from_pdf(pdf_file):
    texts = extract_text_from_pdf(pdf_file)
    assert len(texts) > 0, "Expected text from at least one page"
    for page_text in texts:
        assert page_text.strip(), "Extracted text should not be empty"
