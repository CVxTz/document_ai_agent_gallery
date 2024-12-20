import json
from typing import Literal, Optional

import google.generativeai as genai
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from document_ai_agents.logger import logger
from document_ai_agents.schema_utils import prepare_schema_for_gemini


class AnswerChainOfThoughts(BaseModel):
    rationale: str = Field(
        ...,
        description="Justification of your answer.",
    )
    relevant_context: str = Field(
        ...,
        description="Text snippets from the context that support your answer.",
    )
    answer: str = Field(
        ..., description="Your Answer. Answer with 'N/A' if answer is not found"
    )


class AnswerReformulation(BaseModel):
    declarative_answer: str = Field(
        ..., description="Your Answer. Answer with 'N/A' if answer is not found"
    )


class VerificationChainOfThoughts(BaseModel):
    rationale: str = Field(
        ...,
        description="Justification of your answers.",
    )
    entailment: Literal["Yes", "No"] = Field(
        ...,
        description="Answer 'Yes' if the context entails the assertion and 'No' otherwise",
    )


class DocumentQAState(BaseModel):
    question: str
    pages_as_base64_jpeg_images: list[str] = Field(..., default_factory=list)
    pages_as_text: list[str] = Field(..., default_factory=list)
    answer_cot: Optional[AnswerChainOfThoughts] = None
    answer_reformulation: Optional[AnswerReformulation] = None
    verification_cot: Optional[VerificationChainOfThoughts] = None


class DocumentQAAgent:
    def __init__(self, model_name="gemini-1.5-flash-8b"):
        self.answer_cot_schema = prepare_schema_for_gemini(AnswerChainOfThoughts)
        self.declarative_answer_schema = prepare_schema_for_gemini(AnswerReformulation)
        self.verification_cot_schema = prepare_schema_for_gemini(
            VerificationChainOfThoughts
        )
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            self.model_name,
        )

        self.graph = None
        self.build_agent()

    def answer_question(self, state: DocumentQAState):
        logger.info(f"Responding to question '{state.question}'")
        assert (
            state.pages_as_base64_jpeg_images or state.pages_as_text
        ), "Input text or images"
        messages = [
            {
                "role": "user",
                "parts": [
                    {"mime_type": "image/jpeg", "data": base64_jpeg}
                    for base64_jpeg in state.pages_as_base64_jpeg_images
                ]
                + state.pages_as_text
                + [{"text": state.question}]
                + [
                    {
                        "text": f"Use this schema for your answer: {self.answer_cot_schema}"
                    }
                ],
            }
        ]

        response = self.model.generate_content(
            messages,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": self.answer_cot_schema,
                "temperature": 0.0,
            },
        )

        answer_cot = AnswerChainOfThoughts(**json.loads(response.text))

        return {"answer_cot": answer_cot}

    def reformulate_answer(self, state: DocumentQAState):
        logger.info("Reformulating answer")
        if state.answer_cot.answer == "N/A":
            return

        messages = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": "Reformulate this question and its answer as a single assertion."
                    },
                    {"text": f"Question: {state.question}"},
                    {"text": f"Answer: {state.answer_cot.answer}"},
                ]
                + [
                    {
                        "text": f"Use this schema for your answer: {self.declarative_answer_schema}"
                    }
                ],
            }
        ]

        response = self.model.generate_content(
            messages,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": self.declarative_answer_schema,
                "temperature": 0.0,
            },
        )

        answer_reformulation = AnswerReformulation(**json.loads(response.text))

        return {"answer_reformulation": answer_reformulation}

    def verify_answer(self, state: DocumentQAState):
        logger.info(f"Verifying answer '{state.answer_cot.answer}'")
        if state.answer_cot.answer == "N/A":
            return
        messages = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": "Analyse the following context and the assertion and decide whether the context "
                        "entails the assertion or not."
                    },
                    {"text": f"Context: {state.answer_cot.relevant_context}"},
                    {
                        "text": f"Assertion: {state.answer_reformulation.declarative_answer}"
                    },
                    {
                        "text": f"Use this schema for your answer: {self.verification_cot_schema}. Be Factual."
                    },
                ],
            }
        ]

        response = self.model.generate_content(
            messages,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": self.verification_cot_schema,
                "temperature": 0.0,
            },
        )

        verification_cot = VerificationChainOfThoughts(**json.loads(response.text))

        return {"verification_cot": verification_cot}

    def build_agent(self):
        builder = StateGraph(DocumentQAState)
        builder.add_node("answer_question", self.answer_question)
        builder.add_node("reformulate_answer", self.reformulate_answer)
        builder.add_node("verify_answer", self.verify_answer)

        builder.add_edge(START, "answer_question")
        builder.add_edge("answer_question", "reformulate_answer")
        builder.add_edge("reformulate_answer", "verify_answer")
        builder.add_edge("verify_answer", END)
        self.graph = builder.compile()


if __name__ == "__main__":
    from pathlib import Path

    from document_ai_agents.document_utils import extract_images_from_pdf
    from document_ai_agents.image_utils import pil_image_to_base64_jpeg

    document_path = str(Path(__file__).parents[1] / "data" / "docs.pdf")

    images = extract_images_from_pdf(pdf_path=document_path)
    pages_as_base64_jpeg_images = [pil_image_to_base64_jpeg(x) for x in images]

    _state = DocumentQAState(
        question="What is the highest score on M-RCNN ?",
        pages_as_base64_jpeg_images=pages_as_base64_jpeg_images,
        pages_as_text=[],
    )

    agent = DocumentQAAgent()

    result = agent.graph.invoke(_state)

    print(result["answer_cot"])
    print(result["answer_reformulation"])
    print(result["verification_cot"])

    print(result["answer_cot"].relevant_context)

    assert result["answer_cot"].answer == "0.708"
    assert result["verification_cot"].entailment
