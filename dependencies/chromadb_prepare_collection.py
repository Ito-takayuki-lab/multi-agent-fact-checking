__author__ = "Dong Yihan"

import chromadb

client = chromadb.PersistentClient(path="../chromadb_storage")

collection = client.create_collection(name="fact_storage")

facts = []

