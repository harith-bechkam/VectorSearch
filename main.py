import os
from fastapi import FastAPI, UploadFile, File
from db import connect_milvus
from utils import extract_embedding

from contextlib import asynccontextmanager
collection = None  # global collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    global collection
    collection = connect_milvus()  # now returns a valid collection
    yield
    print("App shutting down.")

app = FastAPI(lifespan=lifespan)

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
def root():
    return {"message": "Hello! Milvus connection is ready."}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    embedding = extract_embedding(file_path)
    print("Embedding length:", len(embedding))

    # Insert into Milvus
    collection.insert([
        [embedding],  # embeddings
        [file.filename]  # filenames
    ])

    collection.load()  # optional: make data queryable immediately
    return {"file": file.filename, "status": "uploaded and embedded"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"Running on port {port} with reload")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
