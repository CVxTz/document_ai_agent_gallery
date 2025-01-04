import json
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import wikipedia
from google.generativeai.protos import FunctionDeclaration
from pydantic import BaseModel, Field

from document_ai_agents.logger import logger

wikipedia.page = lru_cache(maxsize=1024)(
    wikipedia.page
)  # To avoid calling the api twice with the same input


class Tool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.description_and_schema = (
            self.description
            + f"\nTakes as input the following object: {json.dumps(self.input_model.model_json_schema(), indent=4)} "
            + f"\nReturns the following object: {json.dumps(self.output_model.model_json_schema(), indent=4)}"
        )

    @abstractmethod
    def __call__(self, input_data: type(BaseModel)) -> Any:
        pass

    @property
    @abstractmethod
    def input_model(self) -> type(BaseModel):
        pass

    @property
    @abstractmethod
    def output_model(self) -> type(BaseModel):
        pass

    def validate_json(self, json_tool_call: str) -> type(BaseModel):
        """
        Returns an instance of self.input_model
        :param json_tool_call: str json
        :return: an instance of self.input_model
        """
        tool_call_args = json.loads(json_tool_call)
        return self.input_model(**tool_call_args)

    def tool_schema_gemini(self) -> FunctionDeclaration:
        return FunctionDeclaration(
            name=self.name,
            description=self.description_and_schema,
            parameters={
                "properties": self.input_model.model_json_schema()["properties"],
                "required": self.input_model.model_json_schema()["required"],
            },
        )


# Wikipedia Search Tool


class WikipediaSearchQuery(BaseModel):
    search_query: str = Field(
        description="Query to send to wikipedia search, should be short and similar to a wikipedia page title"
    )


class PageSummary(BaseModel):
    page_title: str
    page_summary: str
    page_url: str


class WikipediaSearchResponse(BaseModel):
    page_summaries: list[PageSummary]


class WikipediaSearchTool(Tool):
    def __init__(self, lang="en", max_results=3):
        super().__init__(
            name="WikipediaSearchTool",
            description="Searches through wikipedia pages. ",
        )
        self.lang = lang
        self.max_results = max_results
        wikipedia.set_lang(self.lang)

    def __call__(self, input_data: WikipediaSearchQuery) -> WikipediaSearchResponse:
        titles = wikipedia.search(input_data.search_query, results=self.max_results)
        page_summaries = []
        for title in titles[: self.max_results]:
            try:
                page = wikipedia.page(title=title, auto_suggest=False)
                page_summary = PageSummary(
                    page_title=page.title, page_summary=page.summary, page_url=page.url
                )
                page_summaries.append(page_summary)
            except (wikipedia.DisambiguationError, wikipedia.PageError):
                logger.exception(f"Error getting the page {title=}")

        return WikipediaSearchResponse(page_summaries=page_summaries)

    @property
    def input_model(self) -> type(WikipediaSearchQuery):
        return WikipediaSearchQuery

    @property
    def output_model(self) -> type(WikipediaSearchResponse):
        return WikipediaSearchResponse


if __name__ == "__main__":
    import google.generativeai as genai

    wikipedia_tool = WikipediaSearchTool()

    result = wikipedia_tool(WikipediaSearchQuery(search_query="Stevia"))

    print(result)
    print(WikipediaSearchQuery.model_json_schema())
    print(wikipedia_tool.tool_schema_gemini())

    model = genai.GenerativeModel(
        "gemini-1.5-flash-002",
    )


