from document_ai_agents.document_multi_tool_agent import (
    AgentState,
    ToolCallAgent,
    get_page_content,
    get_wikipedia_page,
    search_duck_duck_go,
    search_wikipedia,
)


def test_agent_invocation():
    agent = ToolCallAgent(
        tools=[
            get_wikipedia_page,
            search_wikipedia,
            search_duck_duck_go,
            get_page_content,
        ]
    )

    initial_state = AgentState(
        messages=[
            {
                "role": "user",
                "parts": [
                    "What is the number and season of the south park episode where they get time traveling immigrants?"
                    "Who was the director of that episode? Where and when was he born? Give me his wikipedia page link."
                ],
            }
        ],
    )

    # Invoke the agent with the initial state
    output_state = agent.graph.invoke(initial_state)

    # Assert that the output state contains messages
    assert "messages" in output_state
    assert len(output_state["messages"]) > 0

    # Assert that the last message contains the expected text
    last_message = output_state["messages"][-1]
    assert "parts" in last_message
    assert len(last_message["parts"]) > 0

    last_part = last_message["parts"][-1]
    assert "text" in last_part

    # Check for specific information in the output
    assert "Trey Parker" in last_part["text"]  # Ensure Trey Parker is mentioned
    assert "1969" in last_part["text"]  # Ensure his birth date is mentioned
