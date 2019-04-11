import os
import uuid
import multiprocessing as mp
from typing import List

import imageio


def try_load_images(from_dir: str, to_dir: str, filenames: List[str]):
    print('try_load_images')
    for filename in filenames:
        print(filename)
        path = os.path.join(from_dir, filename)
        try:
            img = imageio.imread(path, pilmode="RGB")
        except:
            print('rm %s' % path)
            os.remove(path)
        else:
            new_filename = '%s.jpg' % uuid.uuid4().hex
            new_path = os.path.join(to_dir, new_filename)
            imageio.imwrite(new_path, img)


def rm_bad_files_from_gi_images(from_dir: str, to_dir: str):
    """Remove files wich we can't open"""
    print(0)
    filename_list = os.listdir(from_dir)
    filename_chunk_list = list(chunks(filename_list, 100))
    print(1)
    with mp.Pool(mp.cpu_count()) as pool:
        pool.apply_async(
            try_load_images,
            args=(
                from_dir,
                to_dir,
                filename_chunk_list
            )
        )
        # p.map(try_load_images, path_chunk_list)


def rm_by_extension(dirpath: str):
    """Remove files with extension different to jpg/jpeg"""
    path_list = [
        os.path.join(dirpath, filename)
        for filename in os.listdir(dirpath)
    ]
    for path in path_list:
        if path[-4:] != '.jpg' and path[-5:] != '.jpeg':
            os.remove(path)


def chunks(list_, size):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(list_), size):
        yield list_[i:i + size]


if __name__ == '__main__':
    from_dir = '../data/images'
    to_dir = '../data/clear_images'
    rm_bad_files_from_gi_images(from_dir, to_dir)
    # rm_by_extension(dirpath)
