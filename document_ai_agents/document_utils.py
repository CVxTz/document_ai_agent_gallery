import tempfile

from pdf2image import convert_from_bytes
from pypdf import PdfReader

from document_ai_agents.logger import logger


def extract_images_from_pdf(pdf_path: str):
    logger.info(f"Extracting images from PDF: {pdf_path}")
    with open(pdf_path, "rb") as f:
        with tempfile.TemporaryDirectory() as path:
            logger.info(f"Converting PDF to images using temporary directory: {path}")
            images = convert_from_bytes(f.read(), output_folder=path, fmt="jpeg")
            logger.info(f"Extracted {len(images)} images from the PDF.")
            return images


def extract_text_from_pdf(pdf_path: str):
    logger.info(f"Extracting text from PDF: {pdf_path}")
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        logger.info(f"Extracting text from {len(reader.pages)} pages.")
        texts = [page.extract_text() for page in reader.pages]
        logger.info(f"Extracted text from {len(texts)} pages.")
        return texts
