from langchain_core.messages import HumanMessage

from agents.clients import llm_google_client, llm_google_client_8b


def test_llm_google_client():
    response = llm_google_client.invoke([HumanMessage("Respond with OK only.")])

    assert "ok" in response.content.lower()


def test_llm_google_client_8b():
    response = llm_google_client_8b.invoke([HumanMessage("Respond with OK only.")])

    assert "ok" in response.content.lower()
