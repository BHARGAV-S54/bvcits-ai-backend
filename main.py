import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import SummarizeRequest, SummarizeResponse, AnswerRequest, AnswerResponse
from openai import OpenAI
from dotenv import load_dotenv

# Load env vars
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(
    title="BVCITS Chatbot AI Backend",
    description="Summarize and answer based on group messages",
    version="1.0.0",
)

# (Optional) Allow your WordPress domain to call these endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-wordpress-site.com"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):
    """
    Generate a concise summary from a list of messages.
    """
    prompt = (
        "You are a helpful assistant. Summarize the following conversation briefly:\n\n"
        + "\n".join(f"- {m}" for m in req.messages)
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"system","content":prompt}],
            temperature=0.5,
            max_tokens=150,
        )
        summary = resp.choices[0].message.content.strip()
        return SummarizeResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest):
    """
    Answer a question using the entire chat history as context.
    """
    # Construct a prompt that injects history + question
    system_msg = "You are an expert assistant. Use the chat history below to answer the question."
    history_block = "\n".join(f"{i+1}. {m}" for i, m in enumerate(req.history))
    user_msg = f"Question: {req.question}"
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system","content":system_msg},
                {"role":"user","content":history_block + "\n\n" + user_msg}
            ],
            temperature=0.7,
            max_tokens=300,
        )
        answer = resp.choices[0].message.content.strip()
        return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
