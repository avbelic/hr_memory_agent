import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger
from typing import Literal

from src.rag_agent import initialize_rag

from dotenv import load_dotenv
load_dotenv(r'.env')

# Setup logger for LightRAG
setup_logger("lightrag", level="INFO")

# When you launch the project be sure to override the default KG: NetworkX
# by specifying kg="Neo4JStorage".

# Note: Default settings use NetworkX
# Initialize LightRAG with Neo4J implementation.

async def main():
    # Initialize LightRAG
    rag = await initialize_rag()
    try:
        # Define valid modes
        modes: list[Literal['local', 'global', 'hybrid', 'naive', 'mix']] = ['local', 'global', 'hybrid', 'naive', 'mix']
        
        # Perform queries with different search modes
        for mode in modes:
            try:
                result = await rag.aquery(
                    "What promises about Ukraine did Trump make that didn't come true?",
                    param=QueryParam(mode=mode)
                )
                print(f"\nResults for {mode} mode:")
                print(result)
            except Exception as e:
                print(f"Query failed for {mode} mode: {str(e)}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
 

if __name__ == "__main__":
    asyncio.run(main())