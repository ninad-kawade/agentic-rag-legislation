# Local Docker deployment

Requirements: Docker Engine with the Compose plugin (the `docker-compose` command), and one provider key in the shell.

From the project root, run:

```bash
export GOOGLE_API_KEY="your-key"
docker-compose build
docker-compose up
```

The build installs the pinned Python requirements, downloads `BAAI/bge-small-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-6-v2`, and copies the existing ChromaDB and BM25 index into the image. Set `OPENAI_API_KEY` or `GROQ_API_KEY` instead if using those providers. Do not put API keys in the Dockerfile or commit them.

Open the UI at <http://localhost:8501>; the API health check is <http://localhost:8000/health>. The Streamlit session permits 10 questions by default; set `RAG_SESSION_QUESTION_LIMIT` to change it.

Stop both services with `Ctrl+C`, or run `docker-compose down` in another shell.

If `data/index` is missing, rebuild it before building the image with `python -m ingest.build_index`. Docker requires `data/index/bm25.pkl` and `data/index/chroma/` containing the `legislation_chunks` collection.
