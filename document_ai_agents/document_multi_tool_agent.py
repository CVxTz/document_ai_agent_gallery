from operator import add
from typing import Annotated, Callable

import google.generativeai as genai
from google.api_core import retry
from google.generativeai.types import RequestOptions
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from document_ai_agents.logger import logger
from document_ai_agents.tools import (
    get_page_content,
    get_wikipedia_page,
    search_duck_duck_go,
    search_wikipedia,
)


class AgentState(BaseModel):
    messages: Annotated[list, add] = Field(default_factory=list)


class ToolCallAgent:
    def __init__(self, tools: list[Callable], model_name="gemini-2.0-flash-exp"):
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            self.model_name,
            tools=tools,
            system_instruction="You are a helpful agent that has access to different tools. Use them to answer the "
            "user's query if needed. Only use information from external sources that you can cite. "
            "You can use multiple tools before giving the final answer. "
            "If the tool response does not give an adequate response you can use the tools again with different inputs."
            "Only respond when you can cite the source from one of your tools."
            "Only answer I don't know after you have exhausted all ways to use the tools to search for that information.",
        )
        self.tools = tools
        self.tool_mapping = {tool.__name__: tool for tool in self.tools}
        self.graph = None
        self.build_agent()

    def call_llm(self, state: AgentState):
        response = self.model.generate_content(
            state.messages,
            request_options=RequestOptions(
                retry=retry.Retry(initial=10, multiplier=2, maximum=60, timeout=300)
            ),
        )

        return {
            "messages": [
                type(response.candidates[0].content).to_dict(
                    response.candidates[0].content
                )
            ]
        }

    def use_tool(self, state: AgentState):
        assert any("function_call" in part for part in state.messages[-1]["parts"])

        tool_result_parts = []

        for part in state.messages[-1]["parts"]:
            if "function_call" in part:
                name = part["function_call"]["name"]
                func = self.tool_mapping[name]
                result = func(**part["function_call"]["args"])
                tool_result_parts.append(
                    {
                        "function_response": {
                            "name": name,
                            "response": result.model_dump(mode="json"),
                        }
                    }
                )

        return {"messages": [{"role": "tool", "parts": tool_result_parts}]}

    @staticmethod
    def should_we_stop(state: AgentState) -> str:
        logger.debug(
            f"Entering should_we_stop function. Current message: {state.messages[-1]}"
        )  # Added log
        if any("function_call" in part for part in state.messages[-1]["parts"]):
            logger.debug(f"Calling tools: {state.messages[-1]['parts']}")
            return "use_tool"
        else:
            logger.debug("Ending agent invocation")
            return END

    def build_agent(self):
        builder = StateGraph(AgentState)
        builder.add_node("call_llm", self.call_llm)
        builder.add_node("use_tool", self.use_tool)

        builder.add_edge(START, "call_llm")
        builder.add_conditional_edges("call_llm", self.should_we_stop)
        builder.add_edge("use_tool", "call_llm")
        self.graph = builder.compile()


if __name__ == "__main__":
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
                    "Who was the director of that episode? Where and when was he born ? Give me his wikipedia page link."
                ],
            }
        ],
    )

    output_state = agent.graph.invoke(initial_state)

    for message in output_state["messages"]:
        print(message["role"])
        for _part in message["parts"]:
            print(_part)

    print(
        output_state["messages"][-1]["parts"][-1]["text"]
    )  # Trey Parker was born on **October 19, 1969**, in Conifer, Colorado, U.S.

    # initial_state = AgentState(
    #     messages=[
    #         {
    #             "role": "user",
    #             "parts": [
    #                 "Is puffer fish poisonous ? if so, explain why and list some other poisonous (not venomous) fish. Don't cite Wikipedia only."
    #             ],
    #         }
    #     ],
    # )
    #
    # output_state = agent.graph.invoke(initial_state)
    #
    # for message in output_state["messages"]:
    #     print(message["role"])
    #     for _part in message["parts"]:
    #         print(_part)
    #
    # print(
    #     output_state["messages"][-1]["parts"][-1]["text"]
    # )
