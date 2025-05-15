import requests
import json
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON
from rich.syntax import Syntax
from pydantic_ai.messages import ModelMessage, UserPromptPart, TextPart

# Configuration
SESSION_ID = "test_session_1"  # Change this to test different sessions
BASE_URL = "http://localhost:8001"

console = Console()

def test_rag_endpoints():

    # First query
    console.print(Panel.fit("First Query", style="bold green"))
    first_query = {
        "question": "What is the minimal wage in germany?",
        "message_history": []
    }
    
    first_response = requests.post(
        f"{BASE_URL}/query",
        json=first_query,
        params={"session_id": SESSION_ID}
    )
    
    console.print("First Query Response:")
    console.print(Panel(
        first_response.json()['response'],
        title="Agent Response",
        border_style="green"
    ))
    console.print()

    # Second query (follow-up)
    console.print(Panel.fit("Second Query (Follow-up)", style="bold green"))
    second_query = {
        "question": "What was my last question about?",
        "message_history": []
    }
    
    second_response = requests.post(
        f"{BASE_URL}/query",
        json=second_query,
        params={"session_id": SESSION_ID}
    )
    
    console.print("Second Query Response:")
    console.print(Panel(
        second_response.json()['response'],
        title="Agent Response",
        border_style="green"
    ))
    console.print()
    
    # Test message history endpoint
    console.print(Panel.fit(f"Message History for Session: {SESSION_ID}", style="bold yellow"))
    history_response = requests.get(
        f"{BASE_URL}/message-history/{SESSION_ID}"
    )
    
    console.print("Message History:")
    for msg in history_response.json():
        console.print(Panel(
            JSON(json.dumps(msg, indent=2)),
            title="Message",
            border_style="yellow"
        ))
        console.print()

if __name__ == "__main__":
    test_rag_endpoints()