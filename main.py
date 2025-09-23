import os
from fastapi import FastAPI, UploadFile, File, Query
from db import connect_milvus
from utils import extract_embeddings_with_chunks
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager
import time

clip_model = SentenceTransformer("clip-ViT-B-32")
collection_main = None
collection_chunks = None

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global collection_main, collection_chunks
    collection_main, collection_chunks = connect_milvus()
    yield
    print("App shutting down.")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {"message": "Hello! Milvus connection is ready."}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Save file to disk
    with open(file_path, "wb") as f:
        file_bytes = await file.read()
        f.write(file_bytes)
    print(f"Saved file: {file.filename} ({len(file_bytes)} bytes) at {file_path}")

    # Extract embeddings
    full_emb, full_text, chunks = extract_embeddings_with_chunks(file_path)
    print(f"Extracted embedding: full vector dim={len(full_emb)}, chunks={len(chunks)}")

    # Generate unique file_id
    file_id = int(time.time() * 1000)
    print(f"Generated file_id: {file_id}")

    # Insert into main collection
    result_main = collection_main.insert([
        [file_id],
        [file.filename],
        [full_emb],
        [full_text]
    ])
    collection_main.flush()
    print(f"Inserted into main collection: {result_main}")

    # Insert chunks referencing file_id
    if chunks:
        result_chunks = collection_chunks.insert([
            [file_id] * len(chunks),
            [c["chunk_embedding"] for c in chunks],
            [c["chunk_content"] for c in chunks],
            [c["chunk_index"] for c in chunks]
        ])
        collection_chunks.flush()
        print(f"Inserted {len(chunks)} chunks into chunk collection: {result_chunks}")
    else:
        print("No chunks extracted, skipping chunk insert")

    return {
        "file": file.filename,
        "status": "uploaded with chunk embeddings",
        "chunks_count": len(chunks)
    }


# ---------------- Search Endpoint ----------------
@app.get("/search")
async def search(query: str = Query(...), top_k_files: int = 3, top_k_chunks: int = 5):
    query_emb = clip_model.encode(query, convert_to_numpy=True)

    # Search top files
    file_results = collection_main.search(
        data=[query_emb],
        anns_field="full_embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k_files,
        output_fields=["file_name"]
    )

    summary_input = f"Query: {query}\n\n"

    for file_hit in file_results[0]:
        file_id = file_hit.id
        file_name = file_hit.entity.get("file_name")

        # Search top chunks
        chunk_results = collection_chunks.search(
            data=[query_emb],
            anns_field="chunk_embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k_chunks,
            expr=f"file_id == {file_id}",
            output_fields=["content_preview", "chunk_index"]
        )

        summary_input += f"File: {file_name}\n"
        for chunk_hit in chunk_results[0]:
            summary_input += f"Chunk {chunk_hit.entity.get('chunk_index')}: {chunk_hit.entity.get('content_preview')}\n"

    return {"summary_input_for_llm": summary_input}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
