# # Define schema
# from pymilvus import FieldSchema, DataType, CollectionSchema
#
# fields = [
#     FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#     FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),  # 128-dim vector
# ]
#
# schema = CollectionSchema(fields, description="Example collection")
#
# # Create collection
# collection = Collection(name="example_collection", schema=schema)
