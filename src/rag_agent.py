"""Pydantic AI agent that leverages RAG with a local LightRAG and Mem0."""

import os
import sys
import argparse
from dataclasses import dataclass
import asyncio
from typing import Union, AsyncIterator, Iterator, List, Dict, Any, Optional

import dotenv
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from openai import AsyncOpenAI
from mem0 import AsyncMemoryClient

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# Setup logger for LightRAG
setup_logger("lightrag", level="INFO")

# Load environment variables from .env file
dotenv.load_dotenv(r'.env')

WORKING_DIR = "./data"

# if not os.path.exists(WORKING_DIR):
#     os.mkdir(WORKING_DIR)

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please create a .env file with your OpenAI API key or set it in your environment.")
    sys.exit(1)

if not os.getenv("MEM0_API_KEY"):
    print("Error: MEM0_API_KEY environment variable not set.")
    print("Please create a .env file with your Mem0 API key or set it in your environment.")
    sys.exit(1)

# Initialize Async Mem0 client
mem0_client = AsyncMemoryClient(
    api_key=os.getenv("MEM0_API_KEY"),
    org_id=os.getenv("MEM0_ORG_ID"),
    project_id=os.getenv("MEM0_PROJECT_ID")
)

async def initialize_rag():
    rag = LightRAG(
        working_dir="data/",
        llm_model_func=gpt_4o_mini_complete,  # Use gpt_4o_mini_complete LLM model
        embedding_func=openai_embed,
        graph_storage="Neo4JStorage", #<-----------override KG default
        kv_storage="RedisKVStorage",
    )

    # Initialize database connections
    await rag.initialize_storages()
    # Initialize pipeline status for document processing
    await initialize_pipeline_status()

    return rag


@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""
    lightrag: LightRAG
    mem0_client: AsyncMemoryClient
    user_id: str


# Create the Pydantic AI agent
agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=RAGDeps,
    system_prompt=
    """
    You are a helpful assistant that answers different user questions.
    Use the retrieve rag tool to get relevant information from the knowledge base if user input is related to employment policies, labor laws in Germany, HR. 
    Use the retrieve memory tool if user input is related to users personal growth and interests, 
    If the user input is about personal growth or interests, use the store memory tool to store the new memory and inform the user about it.
    If data retrieved from the tools doesn't contain the answer, clearly state that the information isn't available in the stored data and provide your best general knowledge response."""
)


@agent.tool
async def retrieve_rag(context: RunContext[RAGDeps], search_query: str) -> Union[str, AsyncIterator[str]]:
    """Retrieve relevant documents about employment policies, labor laws in Germany, HR from lightRAG knowledge base based on a search query.
    
    Args:
        context: The run context containing dependencies.
        search_query: The search query to find relevant documents.
        
    Returns:
        Formatted context information from the retrieved documents.
    """
    return await context.deps.lightrag.aquery(search_query, param=QueryParam(mode="mix"))


@agent.tool
async def store_memory(context: RunContext[RAGDeps], messages: Union[List[Dict[str, str]], str], user_id: Optional[str] = None) -> Dict[str, Any]:
    """Store a new personal memory in Mem0.
    
    Args:
        context: The run context containing dependencies.
        messages: List of message dictionaries with 'role' and 'content' keys, or just a user message string.
        user_id: Optional user ID to override the default from context.
        
    Returns:
        The result of the memory storage operation.
    """
    user_id = user_id or context.deps.user_id
    result = await context.deps.mem0_client.add(messages, user_id=user_id)
    return result

@agent.tool
async def retrieve_memory(context: RunContext[RAGDeps], query: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search and retrieve personal memories from Mem0.
    
    Args:
        context: The run context containing dependencies.
        query: The search query to find relevant memories.
        user_id: Optional user ID to override the default from context.
        
    Returns:
        List of relevant memories.
    """
    user_id = user_id or context.deps.user_id
    memories = await context.deps.mem0_client.search(query, user_id=user_id)
    return memories

async def run_rag_agent(question: str, user_id: str = "user_andrei") -> str:
    """Run the RAG agent to answer questions.
    
    Args:
        question: The question to answer.
        user_id: The user ID for Mem0 memories.
        
    Returns:
        The agent's response.
    """
    # Create dependencies
    rag = await initialize_rag()
    deps = RAGDeps(lightrag=rag, mem0_client=mem0_client, user_id=user_id)
    
    # Run the agent
    result = await agent.run(question, deps=deps)
    
    return result.output


def main():
    """Main function to parse arguments and run the RAG agent."""
    parser = argparse.ArgumentParser(description="Run a Pydantic AI agent with RAG using Neo4j")
    parser.add_argument("--question", help="The question to answer about")
    
    args = parser.parse_args()
    
    # Run the agent
    response = asyncio.run(run_rag_agent(args.question))
    
    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()
