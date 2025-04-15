from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

app = FastAPI()

# Enable CORS for testing from n8n
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load the embedding model
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
chunker = SemanticChunker(embeddings=embed_model, breakpoint_threshold_type="percentile")

@app.post("/chunk")
async def chunk_text(request: Request):
    data = await request.json()
    docs = chunker.create_documents([data["text"]])
    return [{"content": d.page_content} for d in docs]
