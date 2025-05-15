"""Utility functions for RAG operations."""

import os
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver
from dotenv import load_dotenv
from lightrag import LightRAG
from sklearn.metrics.pairwise import pairwise_distances

# Load environment variables
load_dotenv(r'../.env')

def get_neo4j_driver(database: str = "chunk-entity-relation") -> Driver:
    """Create and return a Neo4j database driver instance.
    
    Args:
        database (str): Name of the Neo4j database to connect to. Defaults to "chunk-entity-relation".
        
    Returns:
        Driver: Neo4j database driver instance
    """
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "password")),
        database=database
    )
    return driver


def get_all_entities(driver: Driver) -> List[Dict[str, Any]]:
    """Get all entities from Neo4j database as a list of dictionaries.
    
    Args:
        driver: Neo4j database driver instance
        
    Returns:
        List[Dict[str, Any]]: List of entity dictionaries containing entity_id, entity_type, and description
    """
    try:
        with driver.session() as session:
            # Query to get all nodes with their properties
            query = """
            MATCH (n)
            RETURN n.entity_id as entity_id,
                   n.entity_type as entity_type,
                   n.description as description
            """
            
            result = session.run(query)
            
            # Convert results to list of dictionaries
            entities = []
            for record in result:
                entity = {
                    "entity_id": record["entity_id"],
                    "entity_type": record["entity_type"],
                    "description": record["description"]
                }
                entities.append(entity)
            
            return entities
            
    except Exception as e:
        print(f"Error getting entities: {str(e)}")
        return []


async def get_embeddings_by_entity_type(rag):
    """Get embeddings from entities_vdb matrix grouped by entity_type.
    
    Returns:
        dict: Dictionary with entity types as keys and values containing:
            - entity_names: list of entity names
            - embeddings: numpy array of corresponding embeddings
            - indices: list of original indices from entities_vdb
    """
    entities_vdb = await rag.entities_vdb.client_storage
    entities_list = await rag.chunk_entity_relation_graph.get_all_labels()
    
    # First, get all entity types and their indices
    entity_types = {}
    for i, ent in enumerate(entities_vdb['data']):
        entity_name = ent['entity_name']
        if entity_name in entities_list:
            node_data = await rag.chunk_entity_relation_graph.get_node(entity_name)
            entity_type = node_data.get('entity_type', 'UNKNOWN')
            if entity_type not in entity_types:
                entity_types[entity_type] = {
                    'entity_names': [],
                    'indices': []
                }
            entity_types[entity_type]['entity_names'].append(entity_name)
            entity_types[entity_type]['indices'].append(i)
    
    # Then create embeddings dictionary with numpy arrays
    embeddings_by_type = {}
    for entity_type, data in entity_types.items():
        indices = data['indices']
        embeddings_by_type[entity_type] = {
            'entity_names': data['entity_names'],
            'embeddings': entities_vdb['matrix'][indices],
            'indices': indices
        }
    
    return embeddings_by_type


async def compute_similarity_metrics(embeddings_by_type: Dict[str, Dict[str, Any]], metric='cosine', threshold=0.8):
    """Compute similarity metrics between embeddings within each entity type group using scikit-learn.
    
    Args:
        embeddings_by_type: Dictionary with entity types as keys and values containing:
            - entity_names: list of entity names
            - embeddings: numpy array of corresponding embeddings
            - indices: list of original indices from entities_vdb
        metric (str): Similarity metric to use. Options include:
            - 'cosine': Cosine similarity
            - 'euclidean': Euclidean distance (converted to similarity)
            - 'manhattan': Manhattan distance (converted to similarity)
            - 'correlation': Correlation coefficient
            - 'jaccard': Jaccard similarity
            - Any other metric supported by sklearn.metrics.pairwise_distances
        threshold (float): Threshold for similarity score (0 to 1)
        
    Returns:
        dict: Dictionary with entity types as keys and values containing:
            - pairs: List of tuples (entity1, entity2, similarity_score) for pairs exceeding threshold
            - similarities: Full similarity matrix as numpy array
            - entity_names: List of entity names in the same order as the similarity matrix
            - metric: The metric used
            - threshold: The threshold used
    """
    results = {}
    
    for entity_type, data in embeddings_by_type.items():
        embeddings = data['embeddings']
        entity_names = data['entity_names']
        
        # Compute pairwise distances
        distances = pairwise_distances(embeddings, metric=metric)
        
        # Convert distances to similarities
        if metric in ['cosine', 'correlation', 'jaccard']:
            similarities = 1 - distances
        else:  # For distance metrics like euclidean, manhattan
            similarities = 1 / (1 + distances)
        
        # Get pairs exceeding threshold (excluding self-similarities)
        pairs = []
        n = len(entity_names)
        for i in range(n):
            for j in range(i+1, n):  # Only compute upper triangle to avoid duplicates
                similarity = similarities[i, j]
                if similarity >= threshold:
                    pairs.append((entity_names[i], entity_names[j], similarity))
        
        # Sort pairs by similarity score in descending order
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        results[entity_type] = {
            'pairs': pairs,
            'similarities': similarities,  # Full similarity matrix
            'entity_names': entity_names,  # Entity names in same order as matrix
            'metric': metric,
            'threshold': threshold
        }
    
    return results

async def merge_similar_entities(rag: LightRAG, similarity_results: Dict[str, Dict[str, Any]], merge_threshold: float = 0.9) -> None:
    """Merge similar entities based on similarity metrics results.
    
    Args:
        rag: LightRAG instance
        similarity_results: Results from compute_similarity_metrics function
        merge_threshold: Threshold for merging entities (default: 0.9)
        
    Returns:
        None
    """
    for entity_type, data in similarity_results.items():
        # Get pairs that exceed the merge threshold
        merge_pairs = [(pair[0], pair[1]) for pair in data['pairs'] if pair[2] >= merge_threshold]
        
        # Process each pair
        for source_entity, target_entity in merge_pairs:
            try:
                # Merge entities using LightRAG's amerge_entities
                await rag.amerge_entities(
                    source_entities=[source_entity],
                    target_entity=target_entity,
                    merge_strategy={
                        "description": "concatenate",  # Combine descriptions
                        "source_id": "join_unique"     # Combine source IDs
                    },
                    target_entity_data={
                        "entity_type": entity_type,
                    }
                )
                print(f"Merged {source_entity} into {target_entity}")
            except Exception as e:
                print(f"Error merging {source_entity} into {target_entity}: {str(e)}")