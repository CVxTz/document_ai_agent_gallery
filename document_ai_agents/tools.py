from functools import lru_cache

import wikipedia
from pydantic import BaseModel

from document_ai_agents.logger import logger

wikipedia.page = lru_cache(maxsize=1024)(
    wikipedia.page
)  # To avoid calling the api twice with the same input

# Search Wikipedia


class PageSummary(BaseModel):
    page_title: str
    page_summary: str
    page_url: str


class WikipediaSearchResponse(BaseModel):
    page_summaries: list[PageSummary]


def search_wikipedia(search_query: str) -> WikipediaSearchResponse:
    """
    Searches through wikipedia pages.
    :param search_query: Query to send to wikipedia search, should be short and similar to a wikipedia page title.
    Search for one item at a time even if it means calling the tool multiple times.
    :return:
    """
    max_results = 5

    titles = wikipedia.search(search_query, results=max_results)
    page_summaries = []
    for title in titles[:max_results]:
        try:
            page = wikipedia.page(title=title, auto_suggest=False)
            page_summary = PageSummary(
                page_title=page.title, page_summary=page.summary, page_url=page.url
            )
            page_summaries.append(page_summary)
        except (wikipedia.DisambiguationError, wikipedia.PageError):
            logger.warning(f"Error getting the page {title=}")

    return WikipediaSearchResponse(page_summaries=page_summaries)


# Get full page


class FullPage(BaseModel):
    page_title: str
    page_url: str
    content: str


def get_wikipedia_page(page_title: str, max_text_size: int = 16_000):
    """
    Gets full content of a wikipedia page
    :param page_title: Make sure this page exists by calling the tool "search_wikipedia" first.
    :param max_text_size: defaults to 16000
    :return:
    """
    try:
        page = wikipedia.page(title=page_title, auto_suggest=False)
        full_page = FullPage(
            page_title=page.title,
            page_url=page.url,
            content=page.content[:max_text_size],
        )
    except (wikipedia.DisambiguationError, wikipedia.PageError):
        logger.warning(f"Error getting the page {page_title=}")
        full_page = FullPage(
            page_title=page_title,
            page_url="",
            content="",
        )

    return full_page


if __name__ == "__main__":
    import google.generativeai as genai

    result = search_wikipedia(search_query="Stevia")

    model = genai.GenerativeModel(
        "gemini-1.5-flash-002",
        tools=[search_wikipedia, get_wikipedia_page],
    )

    response = model.generate_content("What is Stevia ?")

    print(response.candidates[0].content)
    print(type(response.candidates[0].content))
    print(type(response.candidates[0].content).to_dict(response.candidates[0].content))
