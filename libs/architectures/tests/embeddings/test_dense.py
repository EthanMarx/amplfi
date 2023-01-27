import pytest
import torch
from mlpe.architectures.embeddings import (
    DenseEmbedding,
    IncoherentDenseEmbedding,
)


@pytest.fixture(params=[1, 2, 3])
def n_ifos(request):
    return request.param


@pytest.fixture(params=[2048, 8192])
def in_features(request):
    return request.param


@pytest.fixture(
    params=[
        10,
        20,
    ]
)
def kernel_size(request):
    return request.param


@pytest.fixture(params=[128, 256])
def out_features(request):
    return request.param


def test_dense_embedding(n_ifos, in_features, out_features):
    embedding = DenseEmbedding(
        in_features,
        out_features,
        n_ifos,
        hidden_layer_size=100,
        num_hidden_layers=3,
    )
    x = torch.randn(8, n_ifos, in_features)
    y = embedding(x)
    assert y.shape == (8, n_ifos * out_features)

    embedding = IncoherentDenseEmbedding(
        in_features,
        out_features,
        n_ifos,
        hidden_layer_size=100,
        num_hidden_layers=3,
    )
    x = torch.randn(8, n_ifos, in_features)
    y = embedding(x)
    assert y.shape == (8, n_ifos * out_features)
