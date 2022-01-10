import sys

sys.path.insert(1, 'src/models/')

from model import MyAwesomeModel
import torch
import pytest

model = MyAwesomeModel(784, [512, 256], 10, 0.0001)

def test_forward_shape():
    x = torch.rand((8, 28, 28))
    y = model(x)

    assert y.shape == (8, 10) 

def test_bad_input():
    with pytest.raises(ValueError, match="Expected input of shape \(batch_size, 28, 28\)"):
        x = torch.rand((8, 100, 200))
        model(x)

def test_bad_dim():
    with pytest.raises(ValueError, match="Expected input with 3 dimensions"):
        x = torch.rand((8, 100))
        model(x)

@pytest.mark.parametrize("shape", [(1, 28, 28), (10, 28, 28), (100, 28, 28)])
def test_forward_shape_multiple(shape):
    x = torch.rand(shape)
    y = model(x)

    assert y.shape == (shape[0], 10) 