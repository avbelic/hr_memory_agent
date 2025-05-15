import os
import asyncio
from pathlib import Path
from typing import List, Optional
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger
from dotenv import load_dotenv
from rag_agent import initialize_rag

# Load environment variables
load_dotenv(r'.env')

# Setup logger
setup_logger("lightrag", level="INFO")

async def process_text_file(file_path: Path, rag: LightRAG) -> None:
    """Process a single text file and add it to the RAG database."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add the document to RAG
        await rag.ainsert(content)
        print(f"Successfully processed: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

async def ingest_file(file_path: str) -> None:
    """Ingest a single text file into LightRAG."""
    try:
        # Initialize RAG
        rag = await initialize_rag()
        
        # Process the file
        await process_text_file(Path(file_path), rag)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

async def ingest_files(directory: str, file_pattern: str = "*.txt") -> None:
    """Ingest all matching text files from a directory into LightRAG."""
    try:
        # Initialize RAG
        rag = await initialize_rag()
        
        # Get all matching files
        directory_path = Path(directory)
        files = list(directory_path.glob(file_pattern))
        
        if not files:
            print(f"No {file_pattern} files found in {directory}")
            return
        
        # Process files concurrently
        tasks = [process_text_file(file, rag) for file in files]
        await asyncio.gather(*tasks)
        
        print(f"Successfully processed {len(files)} files")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    """Main function to run the ingestion process."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest text files into LightRAG")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Single text file to ingest")
    group.add_argument("--directory", help="Directory containing text files to ingest")
    parser.add_argument("--pattern", default="*.txt", help="File pattern to match (default: *.txt)")
    
    args = parser.parse_args()
    
    # Run the ingestion process
    if args.file:
        asyncio.run(ingest_file(args.file))
    else:
        asyncio.run(ingest_files(args.directory, args.pattern))

if __name__ == "__main__":
    main() 