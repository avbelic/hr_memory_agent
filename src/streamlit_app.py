from dotenv import load_dotenv
import streamlit as st
import asyncio
import os
import json
import websockets
import asyncio
from typing import List, Dict, AsyncGenerator
import nest_asyncio



load_dotenv(r'.env')

# API Configuration
WS_URL = "ws://localhost:8001/ws"
HTTP_URL = "http://localhost:8001"
SESSION_ID = "streamlit_session"

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)          


async def stream_response(question: str) -> AsyncGenerator[str, None]:
    """Stream the response from the WebSocket connection."""
    async with websockets.connect(f"{WS_URL}/{SESSION_ID}") as websocket:
        # Send the question
        await websocket.send(json.dumps({"question": question}))
        
        # Stream the response
        while True:
            try:
                response = await websocket.recv()
                data = json.loads(response)
                
                if data["type"] == "chunk":
                    yield data["content"]
                elif data["type"] == "complete":
                    # Extract just the user question and assistant response
                    for msg in data["new_messages"]:
                        if msg["type"] == "ModelRequest":
                            # Add user question if it has content
                            content = next((part["content"] for part in msg["parts"] if part["part_kind"] == "user-prompt"), None)
                            if content:
                                st.session_state.messages.append({
                                    "role": "user",
                                    "content": content
                                })
                        elif msg["type"] == "ModelResponse":
                            # Add assistant response if it has content
                            content = next((part["content"] for part in msg["parts"] if part["part_kind"] == "text"), None)
                            if content:
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": content
                                })
                    break
                elif data["type"] == "error":
                    st.error(f"Error: {data['content']}")
                    break
                    
            except websockets.exceptions.ConnectionClosed:
                break


def display_message(message):
    """Display a single message in the Streamlit UI."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def main():
    st.title("HR AI assistant")

    # Initialize empty message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    for message in st.session_state.messages:
        display_message(message)

    # Chat input for the user
    user_input = st.chat_input("What do you want to know?")

    if user_input:
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's response
        with st.chat_message("assistant"):
            # Create a placeholder for the streaming text
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            async def process_stream():
                nonlocal full_response
                async for chunk in stream_response(user_input):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
            
            asyncio.run(process_stream())

if __name__ == "__main__":
    main()