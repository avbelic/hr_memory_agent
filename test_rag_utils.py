"""Test script for rag_utils.py functions."""

import os
import asyncio
import pytest
from neo4j import GraphDatabase
from src.rag_utils import get_neo4j_driver, get_all_entities, get_embeddings_by_entity_type, compute_similarity_metrics, merge_similar_entities
from src.rag_agent import initialize_rag

def test_get_all_entities():
    """Test the get_all_entities function to ensure it correctly retrieves entities from Neo4j."""
    # Get Neo4j driver
    driver = get_neo4j_driver()
    
    try:
        # Get entities from the database
        entities = get_all_entities(driver)
        
        # Basic assertions
        assert isinstance(entities, list), "get_all_entities should return a list"
        
        # If there are entities, check their structure
        if entities:
            entity = entities[0]
            assert isinstance(entity, dict), "Each entity should be a dictionary"
            assert "entity_id" in entity, "Entity should have entity_id field"
            assert "entity_type" in entity, "Entity should have entity_type field"
            assert "description" in entity, "Entity should have description field"
            
            # Check data types of fields
            assert isinstance(entity["entity_id"], str), "entity_id should be a string"
            assert isinstance(entity["entity_type"], str), "entity_type should be a string"
            assert isinstance(entity["description"], str), "description should be a string"
            
            print(f"Successfully retrieved {len(entities)} entities")
            print("Sample entity:", entity)
        else:
            print("No entities found in the database")
            
    except Exception as e:
        pytest.fail(f"Test failed with error: {str(e)}")
    finally:
        # Close the driver connection
        driver.close()

async def test_entity_similarity():
    # Initialize RAG
    rag = await initialize_rag()
    
    # Get embeddings by entity type
    print("Getting embeddings by entity type...")
    embeddings_by_type = await get_embeddings_by_entity_type(rag)
    
    # Print entity types found
    print("\nEntity types found:")
    for entity_type in embeddings_by_type.keys():
        print(f"- {entity_type}: {len(embeddings_by_type[entity_type]['entity_names'])} entities")
    
    # Compute similarity metrics
    print("\nComputing similarity metrics...")
    similarity_results = await compute_similarity_metrics(
        embeddings_by_type,
        metric='cosine',
        threshold=0.8  # Adjust this threshold as needed
    )
    
    # Print results for each entity type
    print("\nSimilar entity pairs found:")
    for entity_type, data in similarity_results.items():
        if data['pairs']:
            print(f"\n{entity_type}:")
            for source, target, similarity in data['pairs']:
                print(f"- {source} <-> {target} (similarity: {similarity:.3f})")
        else:
            print(f"\n{entity_type}: No similar pairs found above threshold")

if __name__ == "__main__":
    # Run the test
    # test_get_all_entities() 
    asyncio.run(test_entity_similarity())


