from typing import Literal

from pydantic import BaseModel


class LayoutElement(BaseModel):
    caption: str
    ymin: int
    xmin: int
    ymax: int
    xmax: int
    element_type: Literal["Table", "Figure", "Image"]


class LayoutElements(BaseModel):
    layout_elements: list[LayoutElement] = []


class DocumentLayoutParsingState(BaseModel):
    document_path: str
    pages_as_images: list = []
    pages_as_texts: list = []
    layout_elements: list[LayoutElement] = []


if __name__ == "__main__":
    state = DocumentLayoutParsingState(document_path="A.pdf")

    print(state.document_path)
