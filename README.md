# Evidence RAG

Agentic retrieval-augmented question answering over UK legislation.

The project includes a FastAPI query service, a Streamlit UI, and a local hybrid retrieval index (ChromaDB plus BM25). The Docker image includes the built index and downloads the embedding and reranking models during image build.

## Environment variables

Configure exactly one LLM provider key for the backend. Keys are read from the environment and must never be committed to source control:

- `GOOGLE_API_KEY` for Gemini (recommended for this deployment)
- `OPENAI_API_KEY` for OpenAI
- `GROQ_API_KEY` for Groq

Optional settings are `GOOGLE_CHAT_MODEL`, `RAG_GRADE_MODEL`, and `RAG_GENERATE_MODEL`. The Streamlit per-session question limit defaults to 10 and can be changed with `RAG_SESSION_QUESTION_LIMIT`.

## Local development

Install the pinned packages in `requirements.txt` and build the index with `python -m ingest.build_index` if `data/index` is absent. The standard deployment workflow is in [DEPLOY.md](DEPLOY.md).
