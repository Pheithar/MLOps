import sys
sys.path.insert(1, 'src/data/')

from make_dataset import mnist
trainset, testset = mnist("data/raw", "data/processed")

total_size = len(trainset.dataset) + len(testset.dataset)
batched_size = len(trainset) * trainset.batch_size +\
     len(testset) * testset.batch_size

def test_size():
    assert total_size == batched_size

def test_shape():
    for image, _ in trainset:
        assert image.shape == (8, 28, 28), "Bad shape of train image"

    for image, _ in testset:
        assert image.shape == (8, 28, 28), "Bad shape of test image"

def test_labels():
    for _, labels in trainset:
        for label in labels:
            assert label in range(10), "Bad shape of train label"

    for _, labels in testset:
        for label in labels:
            assert label in range(10), "Bad shape of test label"