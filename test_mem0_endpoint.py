import os
from dotenv import load_dotenv
from mem0 import MemoryClient
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.json import JSON
from rich import print as rprint

# Initialize rich console
console = Console()

# Load environment variables
load_dotenv(r'.env')

# Initialize Mem0 client with API key
api_key = os.getenv("MEM0_API_KEY")
organization_id = os.getenv("MEM0_ORG_ID")
project_id = os.getenv("MEM0_PROJECT_ID")

client = MemoryClient(
    api_key=api_key,
    org_id=organization_id,
    project_id=project_id
)

def test_create_memory():
    """Test creating a new memory"""
    messages = [
        {"role": "user", "content": "I'm looking forward to take the course on project management II"},
        {"role": "assistant", "content": "That's great! I'll remember your career growth goals for project management."}
    ]
    result = client.add(messages, user_id="test_user")
    
    # Create a table for the messages
    table = Table(title="Created Memory Messages")
    table.add_column("Role", style="cyan")
    table.add_column("Content", style="green")
    
    for msg in messages:
        table.add_row(msg["role"], msg["content"])
    
    console.print(Panel(table, title="Memory Creation Result"))
    console.print(Panel(JSON.from_data(result), title="API Response"))
    return result

def test_search_memories():
    """Test searching for memories"""
    query = "What are the user's interests and goals?"
    related_memories = client.search(query, user_id="test_user")
    
    # Create a table for search results
    table = Table(title="Search Results")
    table.add_column("Memory ID", style="cyan")
    table.add_column("Content", style="green")
    table.add_column("Score", style="yellow")
    
    for memory in related_memories:
        table.add_row(
            str(memory.get("id", "N/A")),
            str(memory.get("content", "N/A")),
            str(memory.get("score", "N/A"))
        )
    
    console.print(Panel(table, title=f"Search Results for Query: '{query}'"))
    console.print(Panel(JSON.from_data(related_memories), title="Raw API Response"))
    return related_memories

def test_retrieve_memory():
    """Test retrieving all memories"""
    all_memories = client.get_all(user_id="test_user")
    
    # Create a table for all memories
    table = Table(title="All Memories")
    table.add_column("Memory ID", style="cyan")
    table.add_column("Content", style="green")
    table.add_column("Created At", style="yellow")
    
    for memory in all_memories:
        table.add_row(
            str(memory.get("id", "N/A")),
            str(memory.get("content", "N/A")),
            str(memory.get("created_at", "N/A"))
        )
    
    console.print(Panel(table, title="All Retrieved Memories"))
    console.print(Panel(JSON.from_data(all_memories), title="Raw API Response"))
    return all_memories

if __name__ == "__main__":
    # Run tests
    console.print(Panel.fit("Testing Mem0 API endpoints...", style="bold blue"))
    
    # Create memory
    console.print("\n[bold cyan]1. Testing memory creation:[/bold cyan]")
    memory_result = test_create_memory()
    
    # Search memories
    console.print("\n[bold cyan]2. Testing memory search:[/bold cyan]")
    search_results = test_search_memories()
    
    # Retrieve memories
    console.print("\n[bold cyan]3. Testing memory retrieval:[/bold cyan]")
    all_memories = test_retrieve_memory() 