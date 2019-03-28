from six.moves import urllib
from tensorflow.python.platform import gfile
import os

# CVDF mirror of http://yann.lecun.com/exdb/mnist/
DEFAULT_SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'


def urlretrieve_with_retry(url, filename=None):
    return urllib.request.urlretrieve(url, filename)


def maybe_download(filename, work_directory, source_url):
    if not gfile.Exists(work_directory):
        gfile.MakeDirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not gfile.Exists(filepath):
        temp_file_name, _ = urlretrieve_with_retry(source_url)
        gfile.Copy(temp_file_name, filepath)
        with gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath


# 下载文件
maybe_download(TRAIN_IMAGES, "MNIST_data/", DEFAULT_SOURCE_URL + TRAIN_IMAGES)
maybe_download(TRAIN_LABELS, "MNIST_data/", DEFAULT_SOURCE_URL + TRAIN_LABELS)
maybe_download(TEST_IMAGES, "MNIST_data/", DEFAULT_SOURCE_URL + TEST_IMAGES)
maybe_download(TEST_LABELS, "MNIST_data/", DEFAULT_SOURCE_URL + TEST_LABELS)
