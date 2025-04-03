import time

import numpy as np
import uuid
import pinecone
from pinecone.core.grpc.protos import vector_service_pb2, vector_column_service_pb2
from pinecone.core.utils import _generate_request_id, dict_to_proto_struct, dump_numpy_public, fix_tuple_length, \
    proto_struct_to_dict, load_numpy_public, load_strings_public
from pinecone.core.grpc.protos.vector_service_pb2 import Vector, QueryVector

pinecone.init(api_key='1ae701e5-e8ca-452e-9f93-e79fbc5ebb83', environment='internal-alpha')
ranges = [1000, 10000, 100000,1000000]
d = 256


# pinecone.create_index('benchmark', 256)


def upsert_vs(ids, vecs, metadata):
    index = pinecone.core.grpc.index_grpc.VSIndex('benchmark')
    n = len(ids)
    batch = 200
    for i in range(0, n, batch):
        index.upsert(
            vector_service_pb2.UpsertRequest(vectors=[Vector(id=ids[j], values=vecs[j],
                                                             metadata=metadata[j]) for j in range(i, i + batch)]))


def upsert_vcs(ids, vecs, metadata):
    index = pinecone.core.grpc.index_grpc.CIndex('benchmark')
    n = len(ids)
    batch = 200
    for i in range(0, n, batch):
        index.upsert(
            vector_column_service_pb2.UpsertRequest(ids=ids[i:i + batch], data=dump_numpy_public(vecs[i:i + batch]),
                                                    metadata=metadata[i:i + batch]))


def query_vs(qvec):
    index = pinecone.core.grpc.index_grpc.VSIndex('benchmark')
    query_resp = index.query(
        vector_service_pb2.QueryRequest(queries=[QueryVector(values=qvec)], top_k=10, include_metadata=True,
                                        include_values=True))
    return query_resp

def query_vcs(qvec):
    index = pinecone.core.grpc.index_grpc.CIndex('benchmark')
    query_resp = index.query(
        vector_column_service_pb2.QueryRequest(top_k=10, queries=dump_numpy_public(qvec), include_values=True,
                                               include_metadata=True))
    return query_resp

def test_upsert():
    upsert_times = {}

    for n in ranges:
        vecs = np.random.rand(n, d)
        lvecs = [np.random.rand(d).astype(np.float32).tolist() for i in range(n)]
        ids = [str(uuid.uuid4()) for i in range(n)]
        metadata_choices = [
            {'genre': 'action', 'year': 2020},
            {'genre': 'documentary', 'year': 2021},
            {'genre': 'documentary', 'year': 2005},
            {'genre': 'drama', 'year': 2011},
        ]
        metadata = [metadata_choices[i % len(metadata_choices)] for i in range(n)]
        metadata = [dict_to_proto_struct(m) for m in metadata]
        s = time.perf_counter()
        upsert_vs(ids, lvecs, metadata)
        e = time.perf_counter()
        upsert_times['vs_{}'.format(n)] = e - s
        s = time.perf_counter()
        upsert_vcs(ids, vecs, metadata)
        e = time.perf_counter()
        upsert_times['vcs_{}'.format(n)] = e - s
    print(upsert_times)


def test_query():
    query_times = {}
    qvecs = np.random.rand(10, d)
    lvecs = [np.random.rand(d).astype(np.float32).tolist() for i in range(10)]
    vcs = []
    vs = []
    for i in range(1):
        s = time.perf_counter()
        resp = query_vs(lvecs[i])
        # print(resp)
        e = time.perf_counter()
        vs.append(e - s)
        s = time.perf_counter()
        resp = query_vcs(qvecs[i])
        # print(resp)
        e = time.perf_counter()
        vcs.append(e - s)
    query_times['vs'] = sum(vs) / len(vs)
    query_times['vcs'] = sum(vcs) / len(vcs)
    print(query_times)
