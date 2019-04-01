import numpy as np
from json import dump
from glob import glob
from shutil import copyfile
from os.path import join, basename
from os import remove
import re


def compute_avg_similarity(input_dir, aid):
    """
    Compute the average similarity based on the similarity matrix for the
    given assay id.
    """
    # Copy the targeted dat file to the current directory
    dat_list = glob(join(input_dir, '*.dat'))
    target_dat = [f for f in dat_list if bool(re.search(
        r'dissimilarity_matrix_\d+_\d+_{}.dat'.format(aid),
        f
    ))]

    if len(target_dat) == 0:
        return -1
    else:
        target_dat = target_dat[0]

    base_name = basename(target_dat)
    copyfile(target_dat, base_name)

    # Read the dat map
    length = int(re.sub(r'dissimilarity_matrix_(\d+)_\d+_\d+.dat',
                        r'\1',
                        base_name))
    mat = np.memmap(base_name, dtype='float16', mode='r',
                    shape=(length, length))

    # Need to change mat to float64 to avoid float overflow
    mat = np.array(mat, dtype='float64')
    mat_sum = np.sum(mat) / 2
    pair_num = (length + 1) * length / 2

    # Clean the temp dat file
    remove(base_name)

    return mat_sum / pair_num


if __name__ == '__main__':
    aids = range(212)
    avgs = [-1 for i in aids]
    input_dir = '/mnt/gluster/zwang688/assay_similarity_matrix'
    for a in aids:
        avgs[a] = compute_avg_similarity(input_dir, a)

    dump(avgs, open('avg_similarities.json', 'w'), indent=2)
