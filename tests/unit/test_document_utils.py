from pathlib import Path

from document_ai_agents.document_utils import (
    extract_images_from_pdf,
    extract_text_from_pdf,
)


# Test for extract_images_from_pdf
def test_extract_images_from_pdf():
    pdf_file = Path(__file__).parents[2] / "data" / "docs.pdf"
    images = extract_images_from_pdf(str(pdf_file))
    assert len(images) > 0, "Expected at least one image in the PDF"
    assert all(
        image.format == "JPEG" for image in images
    ), "Images should be in JPEG format"


# Test for extract_text_from_pdf
def test_extract_text_from_pdf():
    pdf_file = Path(__file__).parents[2] / "data" / "docs.pdf"
    texts = extract_text_from_pdf(str(pdf_file))
    assert len(texts) > 0, "Expected text from at least one page"
    for page_text in texts:
        assert page_text.strip(), "Extracted text should not be empty"
