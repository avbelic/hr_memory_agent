from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
import os
from dotenv import load_dotenv
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart
)
from rag_agent import agent, RAGDeps, initialize_rag, mem0_client
from contextlib import asynccontextmanager
from starlette.websockets import WebSocketDisconnect

# Load environment variables
load_dotenv(r'.env')

# Global variable to store the agent dependencies
agent_deps: RAGDeps  # Type hint without initialization

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize lightRAG dependencies and anything else to share across api requests
    global agent_deps
    try:
        rag = await initialize_rag()
        agent_deps = RAGDeps(lightrag=rag, mem0_client=mem0_client, user_id="user_andrei")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize RAG agent: {str(e)}")
    yield
    # Shutdown
    # Add cleanup code here if needed

# Create FastAPI app
app = FastAPI(title="RAG Agent API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str
    message_history: Optional[List[ModelMessage]] = None

class QueryResponse(BaseModel):
    response: str
    new_messages: List[ModelMessage]

# initialize message storage
message_histories: Dict[str, List[ModelMessage]] = {}  # Store message histories by session ID

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest, session_id: str = "default"):
    """
    Query the RAG agent with a question and optional message history.
    """
    try:
        # Get or initialize message history for this session
        if session_id not in message_histories:
            message_histories[session_id] = []
        
        # Run the agent with streaming 
        # FIXME: we may not need to stream the response, we can just return the full response
        async with agent.run_stream(
            request.question,
            deps=agent_deps,
            message_history=message_histories[session_id]
        ) as result:
            full_response = ""
            async for chunk in result.stream_text(delta=True):
                full_response += chunk

            # Update message history
            new_messages = result.new_messages()
            message_histories[session_id].extend(new_messages)

            return QueryResponse(
                response=full_response,
                new_messages=new_messages
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/message-history/{session_id}", response_model=List[ModelMessage])
async def get_message_history(session_id: str = "default"):
    """
    Get the message history for a specific session.
    """
    return message_histories.get(session_id, [])

async def handle_agent_stream(websocket: WebSocket, question: str, session_id: str) -> None:
    """Handle the agent streaming process."""
    # here we are streaming the response to the client
    async with agent.run_stream(
        question,
        deps=agent_deps,
        message_history=message_histories[session_id]
    ) as result:
        # Stream text chunks
        async for chunk in result.stream_text(delta=True):
            await websocket.send_json({
                "type": "chunk",
                "content": chunk
            })
        
        # Send completion message in pure JSON format
        new_messages = [
            {
                "type": msg.__class__.__name__,
                "parts": [
                    {
                        "part_kind": part.part_kind,
                        "content": getattr(part, "content", str(part))
                    } for part in msg.parts
                ]
            } for msg in result.new_messages()
        ]
        
        # Update message history in pure JSON format
        message_histories[session_id].extend(result.new_messages())
        await websocket.send_json({
            "type": "complete",
            "new_messages": new_messages
        })

# WebSocket endpoint for streaming agent responses
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming agent responses."""
    await websocket.accept()
    
    # Initialize session history if needed
    if session_id not in message_histories:
        message_histories[session_id] = []
    
    try:
        while True:
            # Get question from client
            data = await websocket.receive_json()
            question = data.get("question")
            
            if not question:
                await websocket.send_json({
                    "type": "error",
                    "content": "No question provided"
                })
                continue
            
            # Process the question
            try:
                await handle_agent_stream(websocket, question, session_id)
            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "content": str(e)
                })
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "content": str(e)
            })
        except:
            pass  # Connection already closed

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 