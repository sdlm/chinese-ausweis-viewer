import os
import multiprocessing as mp
from typing import List

import imageio


def rm_bad_files_from_gi_images(dirpath: str):
    """Remove files wich we can't open"""
    bad_file_path_list = get_all_bad_filenames(dirpath)
    for path in bad_file_path_list:
        print(path)
        # os.remove(path)


def get_all_bad_filenames(dirpath: str) -> List[str]:
    path_list = [
        os.path.join(dirpath, filename)
        for filename in os.listdir(dirpath)
    ]
    path_chunk_list = list(chunks(path_list, 100))
    with mp.Pool(mp.cpu_count()) as p:
        bad_img_path_chunk_list = p.map(try_load_images, path_chunk_list)

    return [x for p_list in bad_img_path_chunk_list for x in p_list]


def try_load_images(path_list: str) -> List[str]:
    bad_img_path_list = []
    for path in path_list:
        try:
            imageio.imread(path, pilmode="RGB")
        except:
            bad_img_path_list.append(path)
        else:
            continue
    return bad_img_path_list


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
    dirpath = '../data/images'
    rm_bad_files_from_gi_images(dirpath)
    # rm_by_extension(dirpath)
