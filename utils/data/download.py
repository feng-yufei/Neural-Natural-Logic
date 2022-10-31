"""
Download and extract data to the target directory
"""

import argparse
import os
import zipfile
import tarfile


def download(url, target_dir):
    """
    Download data from url and save it in the target directory
    :param url: url to download data from
    :param target_dir: where to put the data
    :return: file_path = target_dir + file_name
    """
    if not os.path.exists(target_dir):
        print("Creating target directory {}...".format(target_dir))
        os.makedirs(target_dir)

    file_path = os.path.join(target_dir, url.split('/')[-1])
    if os.path.exists(file_path):
        print("* Found same data in {} - skipping download...".format(target_dir))
    else:
        print("Downloading data from {}...".format(url))
        file_name = url.split('/')[-1]
        file_path = os.path.join(target_dir, file_name)
        os.system('wget {} -O {}'.format(url, file_path))
    return file_path


def extract(file_path):
    """
    Extract data from a zipped/tar.gz file and delete the archive.
    :param file_path: The archive file's path
    :return:
    """
    dir_path = os.path.dirname(file_path)
    print("Extracting to {}...".format(dir_path))
    # file_name = os.path.split(file_path)[-1]
    if zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path) as zf:
            for name in zf.namelist():
                # Ignore useless files in archives.
                if "__MACOSX" in name or\
                   ".DS_Store" in name or\
                   "Icon" in name:
                    continue
                zf.extract(name, dir_path)
    elif tarfile.is_tarfile(file_path):
        with tarfile.open(file_path) as zf:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(zf, dir_path)
    else:
        raise ValueError('File {} cannot be extracted!', file_path)
    # Delete the archive once the data has been extracted.
    os.remove(file_path)


def download_extract(url, target_dir):
    """
    Download and extract data from url and save it in the target directory.
    :param url: url to download data from
    :param target_dir: where to put the data
    :return:
    """
    file_path = os.path.join(target_dir, url.split('/')[-1])
    target = os.path.join(target_dir,
                          ".".join((url.split('/')[-1]).split('.')[:-1]))

    # Skip download and unzipping if the unzipped data is already available.
    if os.path.exists(target) or os.path.exists(target + ".txt"):
        print("* Found unzipped data in {}, skipping download and unzip..."
              .format(target_dir))
    # Skip downloading if the zipped data is already available.
    elif os.path.exists(file_path):
        print("* Found zipped data in {} - skipping download..."
              .format(target_dir))
        extract(file_path)
    # Download and unzip otherwise.
    else:
        extract(download(url, target_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",
                        help="url of the dataset to download")
    parser.add_argument("--target",
                        default=os.path.join("../..", "data"),
                        help="path to a directory where to save the data")
    args = parser.parse_args()

    download_extract(args.dataset_url, os.path.join(args.target))

