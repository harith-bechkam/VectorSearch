from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

CLOUD_ENDPOINT = "in03-fa4be7617b28e70.serverless.aws-eu-central-1.cloud.zilliz.com"
CLOUD_USER = "db_fa4be7617b28e70"
CLOUD_PASSWORD = "Ut1%TZkmjPfHh49f"

collection_name = "documents"


def connect_milvus():
    """Connect to Milvus cloud and return the collection."""
    connections.connect(
        alias="default",
        host=CLOUD_ENDPOINT,
        port=443,
        user=CLOUD_USER,
        password=CLOUD_PASSWORD,
        secure=True
    )

    # Try to load existing collection
    try:
        collection = Collection(name=collection_name)
        print("Loaded existing collection:", collection_name)
    except Exception:
        # Create new collection if not exists
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="extracted_content", dtype=DataType.VARCHAR, max_length=65535)  # store text content
        ]
        schema = CollectionSchema(fields, description="Document embeddings with content")
        collection = Collection(name=collection_name, schema=schema)
        print("Created new collection:", collection_name)

    # 🔹 Create index if not already created
    try:
        index_params = {
            "index_type": "IVF_FLAT",  # can also use HNSW, IVF_SQ8, etc.
            "metric_type": "COSINE",  # or "L2", "IP"
            "params": {"nlist": 1024}  # number of clusters
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print("Index created on 'embedding'")
    except Exception as e:
        print("Index already exists or failed:", e)

    return collection
