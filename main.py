from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from openai import OpenAI
import os
# import keys
from typing import List, Optional
import logging
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import json
import os
from openai import OpenAI  # or `import openai` depending on your SDK usage

# Load environment variables (optional, for local dev)
from dotenv import load_dotenv
load_dotenv()

# Securely get keys from environment
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Constants
# COLLECTION_NAME = "car_inventory3"
COLLECTION_NAME = "Stratford_Nissan_Knowledge_Base"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"

app = FastAPI()


@app.post("/ask")
async def ask_vapi(request: Request):
    try:
        body = await request.json()
        logger.info(f"Raw request body: {body}")

        # Initialize variables
        tool_call_id = "unknown"
        question = ""

        # Handle Vapi's complex message structure
        if "message" in body:
            # Get the first tool call from either toolCallList or toolCalls
            tool_call = None
            if "toolCallList" in body["message"] and body["message"]["toolCallList"]:
                tool_call = body["message"]["toolCallList"][0]
            elif "toolCalls" in body["message"] and body["message"]["toolCalls"]:
                tool_call = body["message"]["toolCalls"][0]
            
            if tool_call:
                tool_call_id = tool_call.get("id", "unknown")
                
                # Handle both stringified JSON and direct object arguments
                if "function" in tool_call:
                    function_data = tool_call["function"]
                    if "arguments" in function_data:
                        if isinstance(function_data["arguments"], str):
                            try:
                                arguments = json.loads(function_data["arguments"])
                                question = arguments.get("question", "")
                            except json.JSONDecodeError:
                                logger.error("Failed to parse JSON arguments")
                        else:
                            question = function_data["arguments"].get("question", "")
                    
                    # Also check parameters if present
                    if not question and "parameters" in function_data:
                        question = function_data["parameters"].get("question", "")

        logger.info(f"Extracted question: {question}")

        if not question.strip():
            logger.warning("Empty question received")
            return {
                "results": [{
                    "toolCallId": tool_call_id,
                    "result": "Please provide a question about our car inventory or Stretford Nissan"
                }]
            }

        # Rest of your processing logic...
        embedding = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=question
        ).data[0].embedding

        search_results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=5,
            with_payload=True
        )

        context = "\n".join(
            hit.payload.get("text", "") 
            for hit in search_results 
            if hit.payload and "text" in hit.payload
        )

        if not context.strip():
            final_answer = "I couldn't find any matching answer in our knowledge base."
        else:
            llm_response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant having Stratford Nissan Knowledge Base. Only answer from your knowledge base otherwise say you don't have the required info in your knowledge base."
                    },
                    {
                        "role": "user", 
                        "content": f"Context:\n{context}\n\nQuestion:\n{question}"
                    }
                ],
                temperature=0.4,
            )
            final_answer = llm_response.choices[0].message.content.strip()

        return {
            "results": [{
                "toolCallId": tool_call_id,
                "result": final_answer.replace("\n", " ")
            }]
        }

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return {
            "results": [{
                "toolCallId": tool_call_id if 'tool_call_id' in locals() else "unknown",
                "result": "An error occurred while processing your request"
            }]
        }
