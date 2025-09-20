import os
from fastapi import FastAPI
from db import connect_chroma, client  # import connect function and client

app = FastAPI()

# --- Connect to Chroma Cloud at startup ---
connect_chroma()

@app.get("/")
def root():
    return {"message": "Hello!"}

@app.get("/collections")
def list_collections():
    if client:
        return client.list_collections()  # example usage
    return {"error": "DB not connected"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))
    print(f"Running on port {port} with reload")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
