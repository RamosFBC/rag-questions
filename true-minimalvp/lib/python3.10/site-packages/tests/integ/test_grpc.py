# from pinecone import GRPCIndex
import pinecone
import uuid
import numpy as np

pinecone.init(api_key='1ae701e5-e8ca-452e-9f93-e79fbc5ebb83',environmnet='internal-alpha')

index = pinecone.GRPCIndex('topk')

def test_basic():
    print(index.describe_index_stats())
    n = 100
    d = 768
    ids = [str(uuid.uuid4()) for i in range(0, n)]
    vecs = [np.random.rand(d).astype(np.float32).tolist() for i in range(n)]

    ur = index.upsert(ids = ids,vectors = vecs)
    print(ur)

    qvec = [0.1]*768
    qr = index.query(queries=qvec)
    print(qr)

    fr = index.fetch(ids=ids[:2])
    print(fr)

    dr = index.delete(ids=ids[:2])
    print(dr)

    print(index.describe_index_stats())