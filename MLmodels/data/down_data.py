import os

def download_data(data_path='_sjdf',
                  url='http://faculty.washington.edu/rjl/pubs/MLSJdF2021/SJdF_processed_gauge_data.tar.gz',
                  chunk_size=1024):
    """
    Download and extract GeoClaw run data

    """

    from requests import get
    from tqdm import tqdm

    if not os.path.exists('_sjdf'):
        os.mkdir(data_path)

    fname = 'SJdF_processed_gauge_data.tar.gz'
    url_get = get(url, stream=True)
    total_size = 110485878 

    bar_fmt = 'Download{percentage:4.0f}%|{bar}|{n_fmt:>6s}/{total_fmt:>6s}[{elapsed:>6s}<{remaining:>6s}]'

    with open(fname, 'wb') as outfile:
        for data in tqdm(url_get.iter_content(chunk_size=chunk_size),
                         desc=fname,
                         total=int(total_size/chunk_size),
                         bar_format=bar_fmt,
                         unit_scale=False):
            size = outfile.write(data)
    
    # extract tarfile
    from sys import platform

    if ('linux' in platform) or ('darwin' in platform):
        import subprocess

        retcode = subprocess.call(['tar', 'xvzf', fname, '-C', data_path])

    else:
        import tarfile

        tar = tarfile.open(fname)
        tar.extractall(path=data_path)

    # clean up - remove downloaded tarball
    os.remove(fname)


if __name__ == "__main__":

    # download and extract SJdF dataset
    download_data()
