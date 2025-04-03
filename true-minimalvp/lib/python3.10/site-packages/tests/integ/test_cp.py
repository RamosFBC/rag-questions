import time

import pinecone
import uuid
import numpy as np
import random
import sys
import pytest
from pinecone_nuts.utils import vs_adapter
import pinecone
from loguru import logger
from pinecone import QueryVector, Vector, Index, PineconeProtocolError, ApiKeyError, ApiException
from pinecone.core.client import Configuration
# from pinecone.core.client.model.upsert_response import UpsertResponse
# from pinecone import QueryVector
from pinecone.core.grpc.index_grpc import GRPCIndex, GRPCVector, GRPCQueryVector
from pinecone.core.grpc.protos import vector_column_service_pb2
from pinecone.core.utils import dump_numpy_public, dict_to_proto_struct
from pinecone.core.grpc.protos import vector_service_pb2
import cProfile, pstats
from pinecone.core.grpc.protos.vector_column_service_pb2 import NdArray
from google.protobuf.struct_pb2 import Struct

from google.protobuf import json_format
from munch import munchify
import pandas as pd
from pandas import DataFrame


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
# from tests.integration.utils import retry_assert
def do_delete(*ids:list):
    pinecone.init(api_key='a02452e1-75e1-4237-92ff-c595cd76c825', environment='us-west1-gcp')
    index = pinecone.GRPCIndex("del-test")
    index.delete(ids=ids,async_req=True)

def do_upsert(*data):
    pinecone.init(api_key='a02452e1-75e1-4237-92ff-c595cd76c825', environment='us-west1-gcp')
    index = pinecone.GRPCIndex("del-test")
    index.upsert(vectors=data)

import multiprocessing as mp

def test_two_ai():
    pinecone.init(api_key='a02452e1-75e1-4237-92ff-c595cd76c825', environment='us-west1-gcp')
    PINECONE_INDEX = 'test'
    VECTOR_COUNT = 1000000
    VECTOR_DIM = 384
    BATCH_SIZE = 500
    index = pinecone.GRPCIndex(PINECONE_INDEX)
    example_data_generator = map(lambda i: (f'aid-{i}', [random.random() for _ in range(VECTOR_DIM)]),
                                 range(VECTOR_COUNT))
    data = list(example_data_generator)
    with index:
        async_results = [
            index.upsert(vectors=chunk, async_req=True)
            for chunk in chunker(data,BATCH_SIZE)
        ]

        # Wait for and retrieve responses (in case of error)
        # [async_result.result() for async_result in async_results]


def test_fillup():
    n = 10000
    d = 384
    pinecone.init(api_key='a02452e1-75e1-4237-92ff-c595cd76c825', environment='us-west1-gcp')
    index = pinecone.GRPCIndex("test")
    # for i in range(100):

    ids = [str(i)+'b' for i in range(n)]
    vecs = [np.random.rand(d).tolist() for i in range(n)]
    meta = [{'a': 1, 'b': 2, 'c': '3'} for i in range(n)]
    data = tuple(zip(ids, vecs, meta))
    res = []
    for chunk in chunker(data,200):
        res.append(index.upsert(vectors=chunk,async_req=True))
    time.sleep(5)
    for r in res:
        print(r.result())
import uuid

def test_upsert_delete():
    pinecone.init(api_key='a02452e1-75e1-4237-92ff-c595cd76c825', environment='us-west1-gcp')
    index = pinecone.GRPCIndex("del-test")
    n = 200
    d = 512

    for i in range(10000):
        ids = [str(uuid.uuid4()) for j in range(n)]
        vecs = [np.random.rand(d).tolist() for j in range(n)]
        meta = [{'a': 1, 'b': 2, 'c': '3'} for j in range(n)]
        data = tuple(zip(ids, vecs, meta))
        index.upsert(vectors=data)
        index.delete(ids=ids)

    # n = 20000
    # d = 128
    # ids = [str(i) for i in range(n)]
    # vecs = [np.random.rand(d).tolist() for i in range(n)]
    # meta = [{'a': 1, 'b': 2, 'c': '3'} for i in range(n)]
    # data = tuple(zip(ids, vecs, meta))

def test_cl_alpha():
    # pinecone.init(api_key='Jr50Ro7YGGIvRFP8WNEdnNiuLmYpvqdO', environment='internal-alpha')
    # pinecone.init(api_key='2c80b666-82a2-4e24-abd1-15fa467c770c', environment='us-west1-gcp')
    pinecone.init(api_key='7e9bf571-48f5-46c0-8a0f-7069a05ee926', environment='internal-alpha')
    # pinecone.init(api_key='su6f5TsLCvpiIs6PiCxBzbxQmVfvotCT', environment='hirad-dev-gcp')
    # cmp_filter = {"$and": [{"field": {"$eq": "match"}}, {"field": {"$nin": ['v1', 'v2', 'v3']}}]}
    # pinecone.init(api_key='QBSLhAlYRsQA48ydU6iT0VAQnjw4MVCX', environment='dev-yarden')
    for index in pinecone.list_indexes():
        pinecone.delete_index(index)



def test_gong_size():
    # df = pd.read_parquet('/Users/rajat/Downloads/gong.parquet', engine='pyarrow')
    # df = df[:10000]
    # df.to_parquet('gong_small.parquet',engine='pyarrow')
    pinecone.init(api_key='46a90ff7-7ae0-40ec-b6e2-39b0241695fe', environment='gong-poc-us-east1-gcp')
    # pinecone.delete_index("bert-500m")
    # pinecone.delete_index("use-500m")
    # pinecone.create_index("bert-500m", 768, shards=100, pod_type="s1")
    # pinecone.create_index("use-500m", 512, shards=100, pod_type="s1")
    # pinecone.delete_index('upsert-billion-bert')
    # pinecone.create_index('upsert-billion-bert', 768, shards=250,
    #                       index_config={"hybrid": True, "deduplicate": True, "k_bits": 1024})
    index = Index("use-500m")

    # fr = index.fetch(ids=["7090464121407981411-6b94ec6f8bd943e6"],namespace="2110669437875894984")
    # meta = fr['vectors']['7090464121407981411-6b94ec6f8bd943e6']['metadata']
    # print(meta)
    qvec = [0.1]*512
    # print(index.query(queries=[qvec],namespace='1032065396256326366',top_k=10,include_metadata=True))
    print(index.describe_index_stats())
    ds = index.describe_index_stats().namespaces
    vc = 0
    # print(ds)
    for ns in ds:
        # print(ds[ns]['vector_count'])
        vc += ds[ns]['vector_count']
    print(vc)
    print(len(ds))


def test_delete_all():
    pinecone.init(environment="gong-poc-us-east1-gcp", api_key="46a90ff7-7ae0-40ec-b6e2-39b0241695fe")
    for index in pinecone.list_indexes():
        pinecone.delete_index(index)

def test_meme_ai():
    pinecone.init(api_key='d0348e6f-9873-416b-b970-1ac7685b9d0d')
    index =Index("curie-embeddings-3")
    print(pinecone.describe_index("curie-embeddings-3"))
    dquery = [0.1]*4096
    # ds = index.describe_index_stats().namespaces
    # vc = 0
    # # print(ds)
    # for ns in ds:
    #     # print(ds[ns]['vector_count'])
    #     vc += ds[ns]['vector_count']
    # print(vc)
    # print(index.query(queries=[dquery],top_k=5,namespace="search-document",include_metadata=True,include_data=True))

def test_customer():
    # pinecone.init(api_key='e5a254ec-fe3d-49c0-9479-85774003d170', environment='us-west1-gcp')
    # ydc
    # pinecone.init(api_key='d3767b7f-13d5-4560-b539-0a5418bbbab6', environment='us-west1-gcp')
    # two
    # pinecone.init(api_key='7c64c130-f379-4104-a29f-c3f372db69a2', environment='us-west1-gcp')
    # hello retail
    # pinecone.init(api_key='9757ae30-835c-461a-9e97-1afa19d26c32')
    # print(pinecone.list_indexes())

    # searchable
    pinecone.init(api_key='bb1767f7-f95f-4769-8b4d-90ce03572d43', environment='us-east-1-aws')
    # index = pinecone.Index("connector-test")
    index = GRPCIndex('searchable-dev')
    # n = 2
    # print(index.fetch(ids=["a"]))
    f = open("/Users/rajat/pinecone/pinecone-db/ids.txt")

    lines = f.read().splitlines()
    fids = []
    for line in lines:
        fids.append(line.replace('"', ''))

    print(len(fids))
    # for line in lines
    # vecs = [np.random.rand(768).tolist() for i in range(n)]
    # ids = [str(i)*512 for i in range(n)]
    # print(ids)
    # fids = ["c0606b76-13d8-4ccd-be2b-bff24204085e_documentText_0_700"]
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    for chunk in chunker(fids[:2000], 200):
        fr = index.fetch(ids=chunk)
        vecs = fr['vectors']
        for key, val in vecs.items():
            vec_val = val['values']
            try:
                assert len(vec_val) == 384

                for i in vec_val:
                    if not -1 <= i <= 1:
                        print("value "+val['id'])
                        break
            except AssertionError:
                print("length "+val['id']+" "+str(len(vec_val)))
    # for resp in fr:
    #     print(resp)
    #     break
    # qvec = [0.1] * 768
    # print(vecs)
    # data = tuple(zip(ids, vecs))
    # # index.upsert(vectors=data)
    # # pinecone.
    # index.delete(ids=ids)
    # qr = index.query(queries=[[0.1] * 768], top_k=10, include_metadata=True,namespace='example')
    # print(qr)
    # for i in range(4000):
    #     # if i % 500 == 0:
    #     #     time.sleep(1)
    # index = pinecone.GRPCIndex('test-{}'.format(i))
    # print(index.describe_index_stats())
    # ds = index.describe_index_stats().namespaces
    # vc = 0
    # # print(ds)
    # for ns in ds:
    #     # print(ds[ns]['vector_count'])
    #     vc += ds[ns]['vector_count']
    # print(vc)
    # print(len(ds))
    # print(pinecone.describe_index('ion-vectors'))

def test_ch_p():
    pinecone.init(api_key='b3cec885-d405-4358-9618-49331367db5c', environment='internal-beta-aws')
    for index in pinecone.list_indexes():
        i = pinecone.GRPCIndex(index)
        print(index)
        print(i.describe_index_stats())

def test_size():
    pinecone.init(api_key='b3cec885-d405-4358-9618-49331367db5c', environment='internal-beta-aws')
    index = pinecone.GRPCIndex('strongio-4x16')
    print(index.describe_index_stats())
    # pinecone.create_index('strongio-4x16',dimension=2048, pods=4)

def test_scal():
    pinecone.init(api_key='724f029e-db08-427e-b503-c3e8088bf13c', environment='internal-beta')
    # pinecone.create_index("load-test",768,pods=10)
    pinecone.scale_index("load-test",10)
    index = pinecone.GRPCIndex('load-test')
    print(index.describe_index_stats())

def test_grpc():
    pinecone.init(api_key='a02452e1-75e1-4237-92ff-c595cd76c825', environment='us-west1-gcp')
    # pinecone.init(api_key='c26c4ff3-1423-4fc8-b3c9-5cc089baed71', environment='hirad-dev')
    # pinecone.init(api_key='ffca3f13-b041-4112-a6de-49b79b4e288c', environment='internal-beta-aws')
    index = pinecone.GRPCIndex("connector-test")
    print(index.describe_index_stats())
    # print(index.query(queries=[[0.1]*192],top_k=10,namespace="2438433470629494082",include_metadata=True))
    print(index.fetch(ids=["3699457083713742813-2e0b8ac4a7d46cdb"],namespace="2438433470629494082"))
    # index = GRPCIndex("imagenet-example")
    # print(index.query(queries=[[0.1]*1000],top_k=10))
    # pinecone.create_index('connector-test',768,pod_type='s1',pods=10)
    # pinecone.init(api_key='7e9bf571-48f5-46c0-8a0f-7069a05ee926', environment='internal-alpha')
    # pinecone.init(api_key='a02452e1-75e1-4237-92ff-c595cd76c825', environment='us-west1-gcp')
    # pinecone.delete_index('imagenet-example')
    # pinecone.describe_index(lol='what')
    # pinecone.create_index('test', 512)
    # print(pinecone.list_indexes())
    # pinecone.scale_index('test-sst',300)
    # index = Index('kafka-test')
    # # pinecone.create_index(dil='heart')
    # # pinecone.init(api_key='', environment='us-west1-gcp')
    # # pinecone.init(api_key='', environment='us-west1-gcp')
    # # pinecone.list_indexes()
    # # index = Index('test')
    # n = 1500
    # vecs = [np.random.rand(768).tolist() for i in range(n)]
    # ids = ['i'*512 for i in range(n)]
    # meta = [{'yo lo': 1, 'yelo': 2, 'oloy': 3} for i in range(n)]
    # data = tuple(zip(ids, vecs, meta))
    # batch = 400
    #
    # # index = GRPCIndex('test')
    # # print(index.describe_index_stats())
    # # print(index.describe_index_stats())
    # def chunker(seq, size):
    #     return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    #
    # #
    # s = time.perf_counter()
    # res = []
    # for chunk in chunker(data, batch):
    #     index.upsert(vectors=chunk)
    # for chunk in chunker(data, batch):
    #     res.append(index.upsert(vectors=chunk, async_req=True, namespace='ns6'))

    # qvec = [0.1] * 512
    # filter = {'yelo':{'$eq':3}}
    # fr = index.fetch(ids=ids)
    # print(len(fr['vectors']))
    # res = index.query(queries=[qvec], namespace='ns', top_k=1000, include_values=True, include_metadata=True)
    # print(res)
    # rs = [r.result() for r in res]
    # e = time.perf_counter()
    # print(e - s)
    # print(async_results)
    # index.delete(ids=ids,namespace='ns')
    # index.fetch(ids=ids,namespace='ns')
    # print([async_result.result() for async_result in async_results])
    # for chunk in chunker(data, batch):
    #     # res = index.async_upsert(5,chunk)
    #     index.upsert(vectors=chunk)
    # from concurrent.futures import ThreadPoolExecutor
    # with ThreadPoolExecutor(max_workers=10) as executor:
    #     for chunk in chunker(data, batch):
    #         future_results = executor.submit(index.upsert, chunk)
    #     rset= future_results.result()
    # print(res)
    # qr = index.query(queries=[[0.1] * 768], top_k=10, include_metadata=True)
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime')
    # stats.print_stats()
    # index.describe_index_stats(lol=2)
    # index.fetch(ids=['0'], namespace='smtv')



def test_delete():
    pinecone.init(api_key='2c80b666-82a2-4e24-abd1-15fa467c770c', environment='us-west1-gcp')
    # for index in pinecone.list_indexes():
    #     pinecone.delete_index(index)
    print(pinecone.whoami())
