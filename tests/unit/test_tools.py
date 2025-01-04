from pydantic import BaseModel

from document_ai_agents.tools import WikipediaSearchTool


def test_wikipedia_search_tool():
    wikipedia_search_tool = WikipediaSearchTool()

    tool_call_json = '{"search_query": "Stevia"}'
    query = wikipedia_search_tool.validate_json(tool_call_json)

    assert isinstance(query, wikipedia_search_tool.input_model)

    result = wikipedia_search_tool(query)

    assert isinstance(result, BaseModel)
    assert isinstance(result, wikipedia_search_tool.output_model)
