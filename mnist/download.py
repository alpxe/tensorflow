import os
import urllib.request
import sys


def maybe_download(file):
    filepath = os.path.join(dest_directory, file)
    urlpath = os.path.join(DEFAULT_SOURCE_URL, file)

    if not os.path.exists(filepath):  # 如果文件夹中不存在该文件
        # 动态的打印下载进度
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (file, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
            pass

        # 下载文件
        respath, _ = urllib.request.urlretrieve(urlpath, filepath, _progress)
        statinfo = os.stat(respath)  # 获取文件属性信息
        print('Successfully downloaded', file, statinfo.st_size, 'bytes.')
        pass
    pass




dest_directory = "MNIST_DATA/"
DEFAULT_SOURCE_URL = "http://yann.lecun.com/exdb/mnist/" #mnist官方地址

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

if not os.path.exists(dest_directory):  # 如果文件夹不存在
    os.makedirs(dest_directory)  # 创建文件夹
    pass

# 下载
maybe_download(TRAIN_IMAGES)
maybe_download(TRAIN_LABELS)
maybe_download(TEST_IMAGES)
maybe_download(TEST_LABELS)
