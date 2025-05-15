import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, compute_mdhash_id
from typing import Literal, Dict, Any

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


from dotenv import load_dotenv
load_dotenv(r'.env')

async def initialize_rag():
    rag = LightRAG(
        working_dir="data/",
        llm_model_func=gpt_4o_mini_complete,  # Use gpt_4o_mini_complete LLM model
        embedding_func=openai_embed,
        graph_storage="Neo4JStorage", #<-----------override KG default
    )

    # Initialize database connections
    await rag.initialize_storages()
    # Initialize pipeline status for document processing
    # await initialize_pipeline_status()

    return rag


async def main():
    rag = await initialize_rag()
    entities_vdb = await rag.entities_vdb.client_storage
    chunk_vdb = await rag.chunks_vdb.client_storage

    # print(entities_vdb)
    entities_list = await rag.chunk_entity_relation_graph.get_all_labels()
    # print(entities_list)
    target_entity = entities_list[0]
    entity_id = compute_mdhash_id(target_entity, prefix="ent-")
    rel_id = compute_mdhash_id(target_entity, prefix="rel-")
    target_entity_properties = await rag.chunk_entity_relation_graph.get_node(target_entity)
    all_relations = []
    # Get all relationships of the source entities
    edges = await rag.chunk_entity_relation_graph.get_node_edges(target_entity)
    if edges:
        for src, tgt in edges:
            # Ensure src is the current entity
            if src == target_entity:
                edge_data = await rag.chunk_entity_relation_graph.get_edge(
                    src, tgt
                )
                all_relations.append((src, tgt, edge_data, compute_mdhash_id(src + tgt, prefix="rel-")))


    for i, ent in enumerate(entities_vdb['data']):
        if ent['entity_name'] == target_entity:
            print(target_entity)
            print(entity_id)
            print(all_relations)
            print(entities_vdb['data'][i])
            print(entities_vdb['matrix'][i,:])
            source_id = entities_vdb['data'][i]['source_id'] 
            for j, chunk in enumerate(chunk_vdb['data']):
                if chunk['__id__'] == source_id:
                    print(chunk)
                    print(chunk_vdb['matrix'][j,:])


if __name__ == "__main__":
    asyncio.run(main())
