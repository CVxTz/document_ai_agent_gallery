import json
import tempfile

import google.generativeai as genai
import PIL.Image as Image
from pdf2image import convert_from_bytes
from pypdf import PdfReader

from document_ai_agents.states import (
    DetectedLayoutElements,
    DocumentLayoutParsingState,
    LayoutElement,
)
from document_ai_agents.utils import (
    delete_keys_recursive,
    draw_bounding_box_on_image,
    replace_value_in_dict,
)


def load_document(state: DocumentLayoutParsingState):
    with open(state.document_path, "rb") as f:
        with tempfile.TemporaryDirectory() as path:
            images = convert_from_bytes(f.read(), output_folder=path)
            images = [x.convert("RGB") for x in images]

        reader = PdfReader(f)

        texts = [page.extract_text() for page in reader.pages]

        assert len(images) == len(texts)

    return {"pages_as_images": images, "pages_as_texts": texts}


def extract_layout_elements(state: DocumentLayoutParsingState):
    schema = DetectedLayoutElements.model_json_schema()
    schema = replace_value_in_dict(schema.copy(), schema.copy())
    del schema["$defs"]
    delete_keys_recursive(schema, key_to_delete="title")
    delete_keys_recursive(schema, key_to_delete="default")

    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        # Set the `response_mime_type` to output JSON
        # Pass the schema object to the `response_schema` field
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
    )

    layout_elements = []

    for i, image_page in enumerate(state.pages_as_images):
        messages = [
            f"Find and summarize all the relevant layout elements in this pdf page in the following format: "
            f"{DetectedLayoutElements.model_json_schema()}. "
            f"Tables should have at least two columns and at least two rows.",
            image_page,
        ]

        result = model.generate_content(messages)

        data = json.loads(result.text)

        layout_elements += [
            LayoutElement(
                caption=x["caption"],
                ymin=x["ymin"] / 1000,
                xmin=x["xmin"] / 1000,
                ymax=x["ymax"] / 1000,
                xmax=x["xmax"] / 1000,
                page_number=i,
                element_type=x["element_type"],
            )
            for x in data["layout_elements"]
        ]

    return {"layout_elements": layout_elements}


if __name__ == "__main__":
    _state = DocumentLayoutParsingState(
        document_path="/home/youness/PycharmProjects/document_ai_agent_gallery/data/docs.pdf"
    )

    result_node1 = load_document(_state)

    _state.pages_as_images = result_node1["pages_as_images"]

    result_node2 = extract_layout_elements(_state)

    _images: list[Image] = result_node1["pages_as_images"]
    _layout_elements: list[LayoutElement] = result_node2["layout_elements"]

    for layout_element in _layout_elements:
        draw_bounding_box_on_image(
            image=_images[layout_element.page_number],
            ymin=layout_element.ymin,
            ymax=layout_element.ymax,
            xmin=layout_element.xmin,
            xmax=layout_element.xmax,
            display_str_list=(layout_element.element_type,),
        )


    for image in _images:
        image.show()
