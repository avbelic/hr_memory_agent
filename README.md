Simple AI agent capable to assist and adapt to each individual employee as well as answer and update relevant information related to selected HR domains specific to a business.

**Main Functionality**
- this PoC lets users interact with an HR agent server with the agent deciding to either learn new personalized HR relevant information for the user to support its development, HR related interests and concerns (personalized memory), retrieve that personalized information to inform its answer, or retrieve information specific to the HR policies for the specific business (domain relation graph and vector store)
- implemented functionality to integrate new domain policy information into the relation graph and update the vector store
- implemented functionality to deduplicate entities using similarity across embeddings

**Main tech features**

- Agent with several tools based on the 
- [Pydantic AI framework](https://ai.pydantic.dev/)
- [lightRAG](https://github.com/HKUDS/LightRAG) as a lightweight hybrid retrieval and storage system founded on vector store, graph DB and key-value storage
- [Redis](https://github.com/redis/redis) for in-memory cache of llm calls, full docs and document chunks 
- [Mem0](https://github.com/mem0ai/mem0) for individual personalized long-term memory
- [FastAPI](https://fastapi.tiangolo.com/) framework
- Websocket for streaming of LLM responses
- Minimal Streamlit client to interact with the solution
- Fully asynchronous implementation (agent, lightRAG, neo4j, mem0, Streamlit client)
- Message history on server side
- User id and session id for multi user support and session tracking
- basic tests and unit tests for core functionality


[data used for lightRAG (employment policies in Germany)](https://buse.de/wp-content/uploads/2024/01/Employment-Law-in-Germany-digital-Version-2024.pdf)
