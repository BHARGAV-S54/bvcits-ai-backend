from pydantic import BaseModel
from typing import List

class SummarizeRequest(BaseModel):
    messages: List[str]        # List of message texts

class SummarizeResponse(BaseModel):
    summary: str               # Generated summary

class AnswerRequest(BaseModel):
    history: List[str]         # Full chat history
    question: str              # User’s question

class AnswerResponse(BaseModel):
    answer: str                # AI’s answer
