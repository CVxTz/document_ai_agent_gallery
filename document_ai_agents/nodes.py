import json
import tempfile

import google.generativeai as genai
import PIL.Image as Image
from pdf2image import convert_from_bytes
from pypdf import PdfReader

from document_ai_agents.image_utils import (
    draw_bounding_box_on_image,
    pil_image_to_base64_png,
)
from document_ai_agents.logger import logger
from document_ai_agents.schema_utils import (
    delete_keys_recursive,
    replace_value_in_dict,
)
from document_ai_agents.states import (
    DetectedLayoutElements,
    DocumentLayoutParsingState,
    LayoutElement,
)


def load_document(state: DocumentLayoutParsingState):
    logger.info(f"Loading document from: {state.document_path}")
    with open(state.document_path, "rb") as f:
        with tempfile.TemporaryDirectory() as path:
            logger.info(f"Converting PDF to images using temporary directory: {path}")
            images = convert_from_bytes(f.read(), output_folder=path, fmt="png")
            images = [x for x in images]

            reader = PdfReader(f)
            logger.info(f"Extracting text from {len(reader.pages)} pages.")
            texts = [page.extract_text() for page in reader.pages]

            assert len(images) == len(
                texts
            ), "Number of images and text pages mismatch."
            logger.info(f"Successfully loaded {len(images)} pages.")
            return {"pages_as_images": images, "pages_as_texts": texts}


def extract_layout_elements(state: DocumentLayoutParsingState):
    logger.info("Starting layout element extraction.")
    schema = DetectedLayoutElements.model_json_schema()

    schema = replace_value_in_dict(schema.copy(), schema.copy())
    del schema["$defs"]
    delete_keys_recursive(schema, key_to_delete="title")
    delete_keys_recursive(schema, key_to_delete="default")

    model = genai.GenerativeModel(
        "gemini-1.5-flash-002",
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
    )
    logger.info(f"Using Gemini model with schema: {schema}")

    layout_elements = []
    for i, image_page in enumerate(state.pages_as_images):
        logger.info(f"Processing page {i+1}/{len(state.pages_as_images)}")
        messages = [
            f"Find and summarize all the relevant layout elements in this pdf page in the following format: "
            f"{DetectedLayoutElements.model_json_schema()}. "
            f"Tables should have at least two columns and at least two rows. "
            f"The coordinates should overlap with each layout item.",
            {"mime_type": "image/png", "data": pil_image_to_base64_png(image_page)},
        ]

        result = model.generate_content(messages)
        data = json.loads(result.text)
        layout_elements.extend(
            [
                LayoutElement(
                    summary=x["summary"],
                    ymin=x["ymin"] / 1000,
                    xmin=x["xmin"] / 1000,
                    ymax=x["ymax"] / 1000,
                    xmax=x["xmax"] / 1000,
                    page_number=i,
                    element_type=x["element_type"],
                )
                for x in data["layout_elements"]
            ]
        )
        logger.info(
            f"Extracted {len(data['layout_elements'])} layout elements from page {i+1}."
        )

    logger.info(f"Total layout elements extracted: {len(layout_elements)}")
    return {"layout_elements": layout_elements}


if __name__ == "__main__":
    _state = DocumentLayoutParsingState(
        document_path="/home/youness/PycharmProjects/document_ai_agent_gallery/data/doc2.pdf"
    )

    result_node1 = load_document(_state)
    _state.pages_as_images = result_node1["pages_as_images"]
    result_node2 = extract_layout_elements(_state)

    _images: list[Image] = result_node1["pages_as_images"]
    _layout_elements: list[LayoutElement] = result_node2["layout_elements"]

    logger.info("Drawing bounding boxes on images.")
    for layout_element in _layout_elements:
        draw_bounding_box_on_image(
            image=_images[layout_element.page_number],
            ymin=layout_element.ymin,
            ymax=layout_element.ymax,
            xmin=layout_element.xmin,
            xmax=layout_element.xmax,
            display_str_list=(layout_element.element_type,),
        )

    logger.info("Displaying images.")
    for idx, image in enumerate(_images):
        logger.info(f"Displaying image {idx+1}")
        image.show()
