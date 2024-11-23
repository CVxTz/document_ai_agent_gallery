import tempfile

from langchain_core.messages import HumanMessage, SystemMessage
from pdf2image import convert_from_bytes
from pypdf import PdfReader

from document_ai_agents.clients import llm_google_client
from document_ai_agents.logger import logger
from document_ai_agents.states import DocumentLayoutParsingState, LayoutElements
from document_ai_agents.utils import ppm_to_base64_jpeg, delete_keys_recursive, replace_value_in_dict


def load_document(state: DocumentLayoutParsingState):
    logger.info(f"{type(state)=}")

    with open(state.document_path, "rb") as f:
        with tempfile.TemporaryDirectory() as path:
            images = convert_from_bytes(f.read(), output_folder=path)

        reader = PdfReader(f)

        texts = [page.extract_text() for page in reader.pages]

        assert len(images) == len(texts)

    return {"pages_as_images": images, "pages_as_texts": texts}


def extract_layout_elements(state: DocumentLayoutParsingState):
    schema = LayoutElements.model_json_schema()
    schema = replace_value_in_dict(schema.copy(), schema.copy())
    del schema["$defs"]
    # delete_keys_recursive(schema, key_to_delete="title")

    constrained_llm = llm_google_client.with_structured_output(schema)

    layout_elements = []

    for image_page in state.pages_as_images:
        messages = [
            SystemMessage(
                content=f"Find and summarize all the relevant layout elements in this pdf page in the following format: "
                f"{LayoutElements.model_json_schema()}"
            ),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{ppm_to_base64_jpeg(image_page)}"
                        },
                    },
                    {"type": "text", "text": "Find table, figures in this image"},
                ]
            ),
        ]

        result: LayoutElements = constrained_llm.invoke(messages)

        print(result)
        layout_elements += result.layout_elements

    return {"layout_elements": layout_elements}


if __name__ == "__main__":
    _state = DocumentLayoutParsingState(
        document_path="/home/youness/PycharmProjects/document_ai_agent_gallery/data/docs.pdf"
    )

    result_node1 = load_document(_state)

    _state.pages_as_images = result_node1["pages_as_images"]

    result_node2 = extract_layout_elements(_state)

    print(result_node2)
