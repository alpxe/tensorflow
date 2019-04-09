from six.moves import urllib
import tarfile
import os
import sys

dest_directory = "cifar10_data/"
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

# 创建文件夹
if not os.path.exists(dest_directory):  # 如果文件夹不存在
    os.makedirs(dest_directory)  # 创建文件夹

filename = DATA_URL.split("/")[-1]  # 获取文件名 cifar-10-binary.tar.gz
filepath = os.path.join(dest_directory, filename)  # 文件路径

# 下载文件
if not os.path.exists(filepath):  # 如果该文件不存在
    # 动态的打印下载进度
    def _progress(count, block_size, total_size):
        sys.stdout.write(
            '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
        pass


    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    pass

# 解压
extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
print(extracted_dir_path)
if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
