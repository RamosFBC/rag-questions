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
from pinecone.core.grpc.index_grpc import GRPCIndex, GRPCVector, GRPCQueryVector
from concurrent.futures import ThreadPoolExecutor, wait
import time

def test_upsert():
    pinecone.init(api_key='a02452e1-75e1-4237-92ff-c595cd76c825')
    n = 2000
    d = 128
    ids = [str(i) for i in range(n)]
    vecs = [np.random.rand(d).tolist() for i in range(n)]
    meta = [{'a': 1, 'b': 2, 'c': '3'} for i in range(n)]
    data = tuple(zip(ids, vecs, meta))
    batch = 300
    index = GRPCIndex("test")

    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    res = []
    for chunk in chunker(data, batch):
        res.append(index.upsert(vectors=chunk, async_req=False))




def test_fetch():
    pinecone.init(api_key='a02452e1-75e1-4237-92ff-c595cd76c825')
    index = pinecone.GRPCIndex("test")
    n = 100
    ids = [str(i) for i in range(n)]
    res = []
    for i in range(n):
        res.append(index.fetch(ids=ids,async_req=True))
    for r in res:
        print(r.result())

def test_query():
    pinecone.init(api_key='a02452e1-75e1-4237-92ff-c595cd76c825')
    index = pinecone.GRPCIndex("test")
    n = 100
    d = 128
    qvecs = [np.random.rand(d).tolist() for i in range(n)]
    ids = [str(i) for i in range(n)]
    res = []
    for vec in qvecs:
        res.append(index.query(queries=[vec], async_req=False,include_data=True))
    for r in res:
        print(r)


        # print(r)


import concurrent.futures

def copy_future_state(source, destination):
    if source.cancelled():
        destination.cancel()
    if not destination.set_running_or_notify_cancel():
        return
    exception = source.exception()
    if exception is not None:
        destination.set_exception(exception)
    else:
        result = source.result()
        destination.set_result(result)


def chain(pool, future, fn):
    result = concurrent.futures.Future()

    def callback(_):
        try:
            temp = pool.submit(fn, future.result())
            copy = lambda _: copy_future_state(temp, result)
            temp.add_done_callback(copy)
        except:
            result.cancel()
            raise

    future.add_done_callback(callback)
    return result

def wait(seconds):
    return 5*seconds


def to_str(val):
    return 'nice'
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

def test_things():
    pool = ThreadPoolExecutor()
    future1 = pool.submit(wait, 5)
    future2 = chain(pool, future1, to_str)
    # future3 = pool.submit(wait, 10)
    print(future2.result())