import pinecone
from pinecone import GRPCIndex,Index
import pandas as pd
import numpy as np
import json


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def test_load_data_to_df():
    vecs = []
    with open('/Users/rajat/pinecone/pinecone-db/vecs_1.txt') as f:
        for line in f.readlines():
            line = line.replace('[','')
            line = line.replace(']','')
            v = np.fromstring(line,sep=',')
            vecs.append(v.tolist())
    namespace = []
    with open('/Users/rajat/pinecone/pinecone-db/namespace_1.txt') as f:
        for line in f.readlines():
            ns = line.replace('"', "")
            ns = ns.replace("\n", "")
            namespace.append(ns)
    ids = []
    with open('/Users/rajat/pinecone/pinecone-db/ids_1.txt') as f:
        for line in f.readlines():
            id = line.replace('"', "")
            id = id.replace("\n", "")
            ids.append(id)
    metadata = []
    with open('/Users/rajat/pinecone/pinecone-db/metadata_1.txt') as f:
        s_m = f.readlines()
        for s in s_m:
            metadata.append(json.loads(s))
    d = {"vectors":vecs,"ids":ids,"metadata":metadata,"namespace":namespace}
    df = pd.DataFrame(data=d)
    df.to_csv('curie_1.csv',index=False)




def test_insert_from__df():
    df = pd.read_csv("curie_1.csv")
    print(len(df))
    grouped = df.groupby('namespace')
    for name,df_group in grouped:
        vectors = df_group['vectors']
        vecs = []
        for v in vectors:
            line = v.replace('[', '')
            line = line.replace(']', '')
            n = np.fromstring(line, sep=',')
            vecs.append(n.tolist())
        # namespace =  df_group['namespace'].tolist()
        ids =  df_group['ids'].tolist()
        metadata =  df_group['metadata'].tolist()
        metadata = [json.loads(m) for m in metadata]
        pinecone.init(api_key='d0348e6f-9873-416b-b970-1ac7685b9d0d')
        index = GRPCIndex("curie-embeddings-2")
        data = tuple(zip(ids, vecs, metadata))
        for chunk in chunker(data,50):
            index.upsert(vectors=chunk,namespace=name)
    # for chunk1,chunk2 in zip(chunker(data,50),chunker(namespace,50)):
    #     index.upsert(vectors=chunk1, namespace=chunk2[0])
    # for v,n,i,m in zip(vecs,namespace,ids,metadata):
    #     index.upsert(vectors=[(i,v,m)],namespace=n)

def test_do_it():
    test_load_data_to_df()
    test_insert_from__df()