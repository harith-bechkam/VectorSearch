import os
import chromadb

client = None

def connect_chroma():
    global client
    try:
        client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY", "ck-FbbBk3qYPcsYrvAUZ4cUJ4qXUenEH14Vh2f5kfKRLGEp"),
            tenant=os.getenv("CHROMA_TENANT", "bbb94d0d-9c62-4685-b660-9aec1da0cd4d"),
            database=os.getenv("CHROMA_DATABASE", "VectorSearch")
        )
        print("Connected to Chroma Cloud successfully!")
    except Exception as e:
        print(f"Failed to connect to Chroma Cloud: {e}")
        # Optionally exit if DB connection is required
        # import sys; sys.exit(1)
