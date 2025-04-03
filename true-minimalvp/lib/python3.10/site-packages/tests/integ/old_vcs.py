import os
import random
import sys

import pinecone
import pytest
import requests
from loguru import logger
from pinecone import Index, PineconeProtocolError, ApiKeyError, ApiException, GRPCIndex
from pinecone.core.client.model.upsert_response import UpsertResponse
from pinecone.exceptions import PineconeException
from prometheus_client.parser import text_string_to_metric_families

from .remote_index import RemoteIndex, retry_assert

logger.remove()
logger.add(sys.stdout, level=(os.getenv("PINECONE_LOGGING") or "INFO"))

vector_dim = 30
env = os.getenv('PINECONE_ENVIRONMENT')
index_name = os.getenv('PINECONE_INDEX_NAME') or f'test-index-{random.randint(0, 1000)}'


def setup_function():
    api_key = os.getenv('PINECONE_API_KEY')
    pinecone.init(api_key=api_key, environment=env)
    if not bool(os.getenv('PINECONE_INDEX_NAME')):
        pinecone.create_index(index_type='approximated', shards=2, name=index_name,
                              dimension=vector_dim)
        os.environ['PINECONE_INDEX_NAME'] = index_name


@pytest.fixture(scope="module")
def pinecone_init():
    api_key = os.getenv('PINECONE_API_KEY')
    pinecone.init(api_key=api_key, environment=env)


@pytest.fixture(scope='module')
def index(pinecone_init, request):
    if request.param:
        return GRPCIndex(index_name)
    else:
        return Index(index_name)


def get_test_data(vector_count=10, no_meta_vector_count=5):
    """repeatably produces same results for a given vector_count"""
    meta_vector_count = vector_count - no_meta_vector_count
    metadata_choices = [
        {'genre': 'action', 'year': 2020},
        {'genre': 'documentary', 'year': 2021},
        {'genre': 'documentary', 'year': 2005},
        {'genre': 'drama', 'year': 2011},
    ]
    no_meta_vectors = [
        (f'vec{i}', [i / 1000] * vector_dim, None)
        for i in range(no_meta_vector_count)
    ]
    meta_vectors = [
        (f'mvec{i}', [i / 1000] * vector_dim, metadata_choices[i % len(metadata_choices)])
        for i in range(meta_vector_count)
    ]
    return list(meta_vectors) + list(no_meta_vectors)


def get_test_data_dict(vector_count=10, no_meta_vector_count=5):
    return {id: (values, metadata) for id, values, metadata in get_test_data(vector_count, no_meta_vector_count)}


def get_vector_count(index, namespace):
    stats = index.describe_index_stats().namespaces
    if namespace not in stats:
        return 0
    return stats[namespace].vector_count


def write_test_data(index, namespace, vector_count=10, no_meta_vector_count=5):
    """writes vector_count vectors into index, half with metadata half without."""
    data = get_test_data(vector_count, no_meta_vector_count)
    count_before = get_vector_count(index, namespace)
    index.upsert(vectors=data, namespace=namespace)
    retry_assert(lambda: vector_count == get_vector_count(index, namespace) - count_before)


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_summarize_no_api_key(index):
    pinecone.init(api_key='', environment=env)
    with pytest.raises((ApiKeyError, PineconeException)) as exc_info:
        api_response = index.describe_index_stats()
        logger.debug('got api response {}', api_response)
    logger.debug('got expected exception: {}', exc_info.value)


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_summarize_nonexistent_index(pinecone_init, index):
    logger.info("api key header: " + os.getenv('PINECONE_API_KEY'))
    if isinstance(index, GRPCIndex):
        index = GRPCIndex('non-existent-index')
    else:
        index = Index('non-exsitent-index')
    with pytest.raises((PineconeProtocolError, PineconeException)) as exc_info:
        api_response = index.describe_index_stats()
        logger.debug('got api response {}', api_response)
    logger.debug('got expected exception: {}', exc_info.value)


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_invalid_upsert_no_params(index):
    with pytest.raises(TypeError) as exc_info:
        api_response = index.upsert()
        logger.debug('got api response {}', api_response)
    logger.debug('got expected exception: {}', exc_info.value)


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_invalid_upsert_vector_no_values(index):
    with pytest.raises(TypeError) as exc_info:
        api_response = index.upsert(id='id')
        logger.debug('got api response {}', api_response)
    logger.debug('got expected exception: {}', exc_info.value)


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_upsert_vectors_no_metadata(index):
    namespace = 'test_upsert_vectors_no_metadata'
    ids = ['vec1', 'vec2']
    vectors = [[0.1] * vector_dim, [0.2] * vector_dim]
    api_response = index.upsert(vectors=zip(ids, vectors), namespace=namespace)
    assert api_response == UpsertResponse(upserted_count=2)
    logger.debug('got openapi upsert without metadata response: {}', api_response)


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_upsert_vectors(index):
    namespace = 'test_upsert_vectors'
    metadata = [{'genre': 'action', 'year': 2020}, {'genre': 'documentary', 'year': 2021}]
    ids = ['mvec1', 'mvec2']
    values = [[0.1] * vector_dim, [0.2] * vector_dim]
    api_response = index.upsert(vectors=zip(ids, values, metadata), namespace=namespace)
    assert api_response == UpsertResponse(upserted_count=2)
    logger.debug('got openapi upsert with metadata response: {}', api_response)


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_invalid_upsert_vectors_wrong_dimension(index):
    with pytest.raises((ApiException, PineconeException)) as e:
        ids = ['vec1', 'vec2']
        values = [[0.1] * 50, [0.2] * 50]
        api_response = index.upsert(vectors=zip(ids, values), namespace='ns1')
        logger.debug('got api response {}', api_response)
    logger.debug('got expected exception: {}', e.value)
    if isinstance(index, Index):
        assert e.value.status == 400
        assert "Data must be dimension" in str(e.value)
    else:
        assert "Data must be dimension" in str(e.value)


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_fetch_vectors_no_metadata(index):
    namespace = 'test_fetch_vectors_no_metadata'
    vector_count = 10
    write_test_data(index, namespace, vector_count)
    test_data = get_test_data_dict(vector_count)

    api_response = index.fetch(ids=['vec1', 'vec2'], namespace=namespace)
    logger.debug('got openapi fetch without metadata response: {}', api_response)

    assert api_response.vectors.get('vec1')
    assert api_response.vectors.get('vec1').values == test_data.get('vec1')[0]
    assert api_response.vectors.get('vec1').get('metadata') == None


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_fetch_vectors(index):
    namespace = 'test_fetch_vectors'
    vector_count = 10
    write_test_data(index, namespace, vector_count)
    test_data = get_test_data_dict(vector_count)

    api_response = index.fetch(ids=['mvec1', 'mvec2'], namespace=namespace)
    logger.debug('got openapi fetch response: {}', api_response)

    assert api_response.vectors.get('mvec1')
    assert api_response.vectors.get('mvec1').values == test_data.get('mvec1')[0]
    assert api_response.vectors.get('mvec1').metadata == test_data.get('mvec1')[1]


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_fetch_vectors_mixed_metadata(index):
    namespace = 'test_fetch_vectors_mixed_metadata'
    vector_count = 10
    write_test_data(index, namespace, vector_count, no_meta_vector_count=5)
    test_data = get_test_data_dict(vector_count)

    api_response = index.fetch(ids=['vec1', 'mvec2'], namespace=namespace)
    logger.debug('got openapi fetch response: {}', api_response)

    for vector_id in ['mvec2', 'vec1']:
        assert api_response.vectors.get(vector_id)
        assert api_response.vectors.get(vector_id).values == test_data.get(vector_id)[0]
        assert api_response.vectors.get(vector_id).metadata == test_data.get(vector_id)[1]


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_invalid_fetch_nonexistent_vectors(index):
    namespace = 'test_invalid_fetch_nonexistent_vectors'
    write_test_data(index, namespace)

    api_response = index.fetch(ids=['no-such-vec1', 'no-such-vec2'], namespace=namespace)
    logger.debug('got openapi fetch response: {}', api_response)


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_invalid_fetch_nonexistent_namespace(index):
    with pytest.raises((ApiException, PineconeException)) as exc_info:
        api_response = index.fetch(ids=['no-such-vec1', 'no-such-vec2'], namespace='no-such-namespace')
        logger.debug('got openapi fetch response: {}', api_response)
    logger.debug('got expected exception: {}', exc_info.value)


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_summarize(index):
    vector_count = 20
    namespace = 'test_describe_index_stats'
    write_test_data(index, namespace, vector_count=vector_count)

    api_response = index.describe_index_stats()
    logger.debug('got openapi describe_index_stats response: {}', api_response)
    assert len(api_response.namespaces) > 0
    assert api_response.namespaces[namespace].vector_count == vector_count


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_query_simple(index):
    namespace = 'test_query_simple'
    vector_count = 10
    write_test_data(index, namespace, vector_count)
    # simple query - no filter, no data, no metadata
    api_response = index.query(
        queries=[
            [0.1] * vector_dim,
            [0.2] * vector_dim
        ],
        namespace=namespace,
        top_k=10,
        include_values=False,
        include_metadata=False
    )
    logger.debug('got openapi query (no filter, no data, no metadata) response: {}', api_response)

    first_match_vector = api_response.results[0].matches[0]
    assert not first_match_vector.values
    assert not first_match_vector.metadata


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_query_simple_with_values(index):
    namespace = 'test_query_simple_with_values'
    vector_count = 10
    write_test_data(index, namespace, vector_count)
    test_data = get_test_data_dict(vector_count)
    # simple query - no filter, with data, no metadata
    api_response = index.query(
        queries=[
            [0.1] * vector_dim,
            [0.2] * vector_dim
        ],
        namespace=namespace,
        top_k=10,
        include_values=True,
        include_metadata=False
    )
    logger.debug('got openapi query (no filter, with data, no metadata) response: {}', api_response)

    first_match_vector = api_response.results[0].matches[0]
    assert first_match_vector.values == test_data.get(first_match_vector.id)[0]
    assert not first_match_vector.metadata


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_query_simple_with_values_metadata(index):
    namespace = 'test_query_simple_with_values_metadata'
    vector_count = 10
    write_test_data(index, namespace, vector_count)
    test_data = get_test_data_dict(vector_count)
    # simple query - no filter, with data, with metadata
    api_response = index.query(
        queries=[
            [0.1] * vector_dim,
            [0.2] * vector_dim
        ],
        namespace=namespace,
        top_k=10,
        include_values=True,
        include_metadata=True
    )
    logger.debug('got openapi query (no filter, with data, with metadata) response: {}', api_response)

    first_match_vector = api_response.results[0].matches[0]
    assert first_match_vector.values == test_data.get(first_match_vector.id)[0]
    if first_match_vector.id.startswith('mvec'):
        assert first_match_vector.metadata == test_data.get(first_match_vector.id)[1]
    else:
        assert not first_match_vector.metadata


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_query_simple_with_values_mixed_metadata(index):
    namespace = 'test_query_simple_with_values_mixed_metadata'
    top_k = 10
    vector_count = 10
    write_test_data(index, namespace, vector_count, no_meta_vector_count=5)
    test_data = get_test_data_dict(vector_count, no_meta_vector_count=5)
    # simple query - no filter, with data, with metadata
    api_response = index.query(
        queries=[
            [0.1] * vector_dim,
            [0.2] * vector_dim
        ],
        namespace=namespace,
        top_k=top_k,
        include_values=True,
        include_metadata=True
    )
    logger.debug('got openapi query (no filter, with data, with metadata) response: {}', api_response)

    for query_vector_results in api_response.results:
        assert len(query_vector_results.matches) == top_k
        for match_vector in query_vector_results.matches:
            assert match_vector.values == test_data.get(match_vector.id)[0]
            if test_data.get(match_vector.id)[1]:
                assert match_vector.metadata == test_data.get(match_vector.id)[1]
            else:
                assert not match_vector.metadata


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_query_simple_with_filter_values_metadata(index):
    namespace = 'test_query_simple_with_filter_values_metadata'
    vector_count = 10
    write_test_data(index, namespace, vector_count)
    test_data = get_test_data_dict(vector_count)
    # simple query - with filter, with data, with metadata
    api_response = index.query(
        queries=[
            [0.1] * vector_dim,
            [0.2] * vector_dim
        ],
        namespace=namespace,
        top_k=10,
        include_values=True,
        include_metadata=True,
        filter={'genre': {'$in': ['action']}}
    )
    logger.debug('got openapi query (with filter, with data, with metadata) response: {}', api_response)

    first_match_vector = api_response.results[0].matches[0]
    assert first_match_vector.values == test_data.get(first_match_vector.id)[0]
    assert first_match_vector.metadata == test_data.get(first_match_vector.id)[1]
    assert first_match_vector.metadata.get('genre') == 'action'


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_query_mixed_metadata_sanity(index):
    namespace = 'test_query_mixed_metadata'
    vectors = [('1', [0.1] * vector_dim, {'colors': 'yellow'}),
               ('2', [-0.1] * vector_dim, {'colors': 'red'})]
    upsert_response = index.upsert(vectors=vectors, namespace=namespace)
    logger.debug('got upsert response: {}', upsert_response)

    query1_response = index.query(queries=[([0.1] * vector_dim, {'colors': 'yellow'})],
                                  top_k=10,
                                  include_metadata=True,
                                  namespace=namespace)
    logger.debug('got first query response: {}', query1_response)

    query2_response = index.query(queries=[([0.1] * vector_dim, {}), ([0.1] * vector_dim, {'colors': 'yellow'})],
                                  top_k=10,
                                  include_metadata=True,
                                  namespace=namespace)
    logger.debug('got second query response: {}', query2_response)

    vectors_dict = {k: m for k, _, m in vectors}
    for query_vector_results in query1_response.results:
        assert len(query_vector_results.matches) == 1
        for match_vector in query_vector_results.matches:
            if vectors_dict.get(match_vector.id):
                assert match_vector.metadata == vectors_dict.get(match_vector.id)
            else:
                assert not match_vector.metadata

    for query_vector_results in query2_response.results:
        for match_vector in query_vector_results.matches:
            if vectors_dict.get(match_vector.id):
                assert match_vector.metadata == vectors_dict.get(match_vector.id)
            else:
                assert not match_vector.metadata


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_invalid_query_nonexistent_namespace(index):
    with pytest.raises((ApiException, PineconeException)) as exc_info:
        api_response = index.query(
            queries=[
                [0.1] * vector_dim,
                [0.2] * vector_dim
            ],
            namespace='no-such-ns',
            top_k=10,
            include_values=True,
            include_metadata=True,
            filter={'action': {'$in': ['action']}}
        )
        logger.debug('got openapi query (with filter, with data, with metadata) response: {}', api_response)
    logger.debug('got expected exception: {}', exc_info.value)


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_query_with_per_vector_top_k(index):
    namespace = 'test_query_with_per_vector_top_k'
    write_test_data(index, namespace)
    # query with query-vector-specific top_k override
    api_response = index.query(
        queries=[([0.1] * vector_dim, {}), ([0.2] * vector_dim, {})],
        namespace=namespace,
        top_k=10,
        include_values=True,
        include_metadata=True
    )
    logger.debug('got openapi query response: {}', api_response)


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_query_uses_distributed_knn(index):
    namespace = 'test_query_with_multi_shard'
    write_test_data(index, namespace, vector_count=1000, no_meta_vector_count=1000)
    query_response = index.query(
        queries=[
            [0.1] * vector_dim,
            [0.2] * vector_dim,
        ],
        namespace=namespace,
        top_k=500,
        include_values=False,
        include_metadata=False
    )
    # assert that we got the same number of results as the top_k
    # regardless of the number of shards
    for query_vector_results in query_response.results:
        assert len(query_vector_results.matches) == 500
    logger.debug('got openapi query response: {}', query_response)


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_delete(index):
    namespace = 'test_delete'
    vector_count = 10
    write_test_data(index, namespace, vector_count)
    test_data = get_test_data_dict(vector_count)

    api_response = index.fetch(ids=['mvec1', 'mvec2'], namespace=namespace)
    logger.debug('got openapi fetch response: {}', api_response)
    assert api_response.vectors and api_response.vectors.get('mvec1').values == test_data.get('mvec1')[0]

    vector_count = get_vector_count(index, namespace)
    api_response = index.delete(ids=['vec1', 'vec2'], namespace=namespace)
    logger.debug('got openapi delete response: {}', api_response)
    retry_assert(lambda: get_vector_count(index, namespace) == (vector_count - 2))
    api_response = index.fetch(ids=['no-such-vec1', 'no-such-vec2'], namespace=namespace)
    logger.debug('got openapi fetch response: {}', api_response)


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_delete_all(index):
    namespace = 'test_delete_all'
    write_test_data(index, namespace)
    api_response = index.delete(delete_all=True, namespace=namespace)
    logger.debug('got openapi delete response: {}', api_response)


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_invalid_delete_nonexistent_ids(index):
    namespace = 'test_nonexistent_ids'
    write_test_data(index, namespace)
    api_response = index.delete(ids=['no-such-vec-1', 'no-such-vec-2'], namespace=namespace)
    logger.debug('got openapi delete response: {}', api_response)


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_invalid_delete_from_nonexistent_namespace(index):
    namespace = 'test_delete_namespace_non_existent'
    with pytest.raises((ApiException, PineconeException)) as exc_info:
        api_response = index.delete(ids=['vec1', 'vec2'], namespace=namespace)
        logger.debug('got openapi delete response: {}', api_response)
    logger.debug('got expected exception: {}', exc_info.value)


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_delete_all_nonexistent_namespace(index):
    namespace = 'test_delete_all_non_existent'
    with pytest.raises((ApiException, PineconeException)) as exc_info:
        api_response = index.delete(delete_all=True, namespace=namespace)
        logger.debug('got openapi delete response: {}', api_response)
    logger.debug('got expected exception: {}', exc_info.value)


@pytest.mark.parametrize('index', [0, 1], indirect=True)
def test_scrape(index):
    api_key = os.getenv('PINECONE_API_KEY')
    endpoint = f'https://metrics.{env}.pinecone.io/metrics'
    r = requests.get(endpoint, headers={'Authorization': f'Bearer {api_key}'})
    # the response status code should've been 2xx (if not, this raises an Error):
    r.raise_for_status()
    # the response should contain prometheus metrics:
    assert len(list(text_string_to_metric_families(r.text))) > 0


def test_cleanup():
    """Cleanup a testing directory once we are finished."""
    for index in pinecone.list_indexes():
        pinecone.delete_index(index, timeout=300)
    current_indexes = pinecone.list_indexes()
    assert current_indexes == []
