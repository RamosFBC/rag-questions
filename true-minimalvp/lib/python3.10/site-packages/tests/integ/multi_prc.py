import multiprocessing
import os
import time
import pinecone


def upsert(data, prc, index_name, batch, return_dict):
    pinecone.init(api_key='076a7136-9e84-45d0-b802-bd33861c5dc8', environment='gong-poc-us-east1-gcp')
    index = pinecone.Index(index_name)

    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    start = time.perf_counter()
    for chunk in chunker(data, batch):
        index.upsert(vectors=chunk)
    end = time.perf_counter()
    diff = end - start
    return_dict[prc] = diff


import numpy as np
import uuid
import sys, argparse
from pinecone.core.grpc import index_grpc
from pinecone.core.grpc.protos import vector_column_service_pb2
from pinecone.core.utils import dump_numpy_public


def upsert_grpc(data, prc, index_name, batch, return_dict):
    key = '2c80b666-82a2-4e24-abd1-15fa467c770c'
    pinecone.init(api_key=key, environment='us-west1-gcp')
    index = pinecone.Index(index_name)

    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    start = time.perf_counter()
    for chunk in chunker(data, batch):
        index.upsert(vectors=chunk)
    end = time.perf_counter()
    diff = end - start
    return_dict[prc] = diff

from pinecone.core.grpc.protos import vector_service_pb2
def upsert_c(vecs,ids,metadata,index_name,batch):
    key = '2c80b666-82a2-4e24-abd1-15fa467c770c'
    pinecone.init(api_key=key, environment='us-west1-gcp')
    index = pinecone.core.grpc.index_grpc.CIndex(index_name)
    n = len(ids)
    start = time.perf_counter()
    for i in range(0,n,batch):
        index.upsert(vector_column_service_pb2.UpsertRequest(ids=ids[i:i+batch],data=dump_numpy_public(vecs[i:i+batch]),metadata=metadata[i:i+batch]))
    end = time.perf_counter()
    diff = end-start
    print(diff)
    index.describe_index_stats(vector_column_service_pb2.DescribeIndexStatsRequest())
from itertools import cycle

# if __name__ == "__main__":
def test_stuff():
    pinecone.init(api_key='2c80b666-82a2-4e24-abd1-15fa467c770c', environment='us-west1-gcp')
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--n', type=int, default=100000)
    # parser.add_argument('--num_prc', type=int, default=10)
    # parser.add_argument('--include_metadata', type=int, default=1)
    # parser.add_argument('--meta_size', type=int, default=30)
    # parser.add_argument('--sharded', type=bool, default=True)
    # parser.add_argument('--batch_size', type=int, default=300)
    # parser.add_argument('--grpc', type=int, default=1)
    mgr = multiprocessing.Manager()
    return_dict = mgr.dict()
    # args = parser.parse_args()
    # print(args)
    # n = args.n
    # pinecone.create_index('upsert-test',768)
    n = 10000
    # num_prc = args.num_prc
    num_prc =1
    vectors_per_process = int(n / num_prc)
    print('Vectors per process', vectors_per_process)
    # include_meta = args.include_metadata
    include_meta = 1
    # m = args.meta_size
    m = 25
    # sharded = args.sharded
    sharded = 1
    # batch = args.batch_size
    batch = 100
    d = 768
    # grpc = args.grpc
    grpc = 0
    # parse args
    weather_vocab = ['sunny', 'rain', 'cloudy', 'snowy']
    loop = cycle(weather_vocab)
    meta_dict = {'mkey{}'.format(i): i for i in range(m)}
    print('metadata size:', sys.getsizeof(meta_dict))
    metadata = [meta_dict for i in range(n)]
    ids = [str(uuid.uuid4()) for i in range(0, n)]
    vectors = [np.random.rand(d).astype(np.float32).tolist() for i in range(n)]
    if (include_meta):
        data = tuple(zip(ids, vectors, metadata))
    else:
        data = tuple(zip(ids, vectors))
    if sharded:
        index_name = 'upsert-test'
    else:
        index_name = 'upsert_test-1'
    jobs = []
    start = time.perf_counter()
    for i in range(num_prc):
        if grpc:
            p = multiprocessing.Process(target=upsert, args=(
                data[i * vectors_per_process:(i + 1) * vectors_per_process], i, index_name, batch, return_dict))
        else:
            p = multiprocessing.Process(target=upsert, args=(
        data[i * vectors_per_process:(i + 1) * vectors_per_process], i, index_name, batch, return_dict))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    end = time.perf_counter()
    diff = end - start
    print('time take by all processes  ', diff)
    print('time taken by individual processes ', return_dict.values())