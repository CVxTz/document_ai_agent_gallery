from functools import lru_cache, wraps
from typing import Any, Callable

import requests
import wikipedia
from duckduckgo_search import DDGS
from pydantic import BaseModel
from strip_tags import strip_tags

from document_ai_agents.logger import logger

wikipedia.page = lru_cache(maxsize=1024)(
    wikipedia.page
)  # To avoid calling the api twice with the same input


class ErrorResponse(BaseModel):
    error: str
    success: bool = False


def catch_exceptions(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            # Call the original function
            response = func(*args, **kwargs)
            return response
        except Exception as e:
            logger.warning(f"Function call failed: {func} {e}")
            return ErrorResponse(error=str(e))

    return wrapper


# Search Wikipedia


class PageSummary(BaseModel):
    page_title: str
    page_summary: str
    page_url: str


class SearchResponse(BaseModel):
    page_summaries: list[PageSummary]


@catch_exceptions
def search_wikipedia(search_query: str) -> SearchResponse:
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

    return SearchResponse(page_summaries=page_summaries)


# Get full page


class FullPage(BaseModel):
    page_title: str
    page_url: str
    content: str


@catch_exceptions
def get_wikipedia_page(page_title: str, max_text_size: int = 16_000):
    """
    Gets full content of a wikipedia page
    :param page_title: Make sure this page exists by calling the tool "search_wikipedia" first.
    :param max_text_size: defaults to 16000
    :return:
    """
    try:
        page = wikipedia.page(title=page_title, auto_suggest=False)
        full_content = strip_tags(page.html())
        full_page = FullPage(
            page_title=page.title,
            page_url=page.url,
            content=full_content[:max_text_size],
        )
    except (wikipedia.DisambiguationError, wikipedia.PageError):
        logger.warning(f"Error getting the page {page_title=}")
        full_page = FullPage(
            page_title=page_title,
            page_url="",
            content="",
        )

    return full_page


# DuckDuckGo search


@catch_exceptions
def search_duck_duck_go(search_query: str) -> SearchResponse:
    """
    Searches through duckduckgo pages.
    :param search_query: Query to send to DuckDuckGo search.
    Search for one item at a time even if it means calling the tool multiple times.
    :return:
    """
    max_results = 10

    with DDGS() as dd:
        results_generator = dd.text(
            search_query,
            max_results=max_results,
            backend="api",
        )

    return SearchResponse(
        page_summaries=[
            PageSummary(
                page_title=x["title"], page_summary=x["body"], page_url=x["href"]
            )
            for x in results_generator
        ]
    )


# Get page content


@catch_exceptions
def get_page_content(page_title: str, page_url: str) -> FullPage:
    """
    Gets page content
    :param page_title: Page title.
    :param page_url: Url to use.
    :return: FullPage object containing page title, URL, and content.
    """
    # Fetch the HTML content of the page
    response = requests.get(page_url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Extract the HTML content
    html = response.text

    # Strip HTML tags to get plain text content
    content = strip_tags(html)

    content = "\n".join([x for x in content.split("\n") if x.strip()])

    return FullPage(
        page_title=page_title,
        page_url=page_url,
        content=content,
    )


if __name__ == "__main__":
    # import google.generativeai as genai
    #
    # result = search_wikipedia(search_query="Stevia")
    #
    # model = genai.GenerativeModel(
    #     "gemini-1.5-flash-002",
    #     tools=[search_wikipedia, get_wikipedia_page],
    # )
    #
    # response = model.generate_content("What is Stevia ?")
    #
    # print(response.candidates[0].content)
    # print(type(response.candidates[0].content))
    # print(type(response.candidates[0].content).to_dict(response.candidates[0].content))

    result = search_duck_duck_go(search_query="Stevia")
    print(result)

    print(
        get_page_content(
            page_title="Stevia", page_url="https://en.wikipedia.org/wiki/Stevia"
        )
    )
