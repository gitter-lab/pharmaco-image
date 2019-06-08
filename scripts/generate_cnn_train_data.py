import numpy as np
import re
import cv2
import os
import zipfile
import time
from sys import argv
from glob import glob
from os.path import join, basename, exists
from os import mkdir
from shutil import rmtree
from json import load

# Global variables
raw_channels = ['ERSyto', 'ERSytoBleed', 'Hoechst', 'Mito', 'Ph_golgi']
raw_paths = ['./{}-{}/*.tif'.format('{}', c) for c in raw_channels]


def select_wells(assay):
    """
    Given an assay id, select pid and wid to extract images from.

    Args:
        assay(int): assay index (column of the output matrix)

    Return:
        a dictionary: {pid: [(wid, label), (wid, label)...]}
    """

    # Load the output matrix and each row's corresponding pid, wid
    output_data = np.load('./output_matrix_convert_collision.npz',
                          allow_pickle=True)
    output_matrix = output_data['output_matrix']
    pid_wids = output_data['pid_wids']

    # Find selected compounds in this assay
    selected_index = output_matrix[:, assay] != -1
    selected_labels = output_matrix[:, assay][selected_index]
    selected_pid_wids = np.array(pid_wids)[selected_index]

    # Flatten the selected pid_wids and group them by pid
    # selected_wells has structure [(wid, pid, label)]
    selected_wells = []

    for i in range(len(selected_pid_wids)):
        cur_pid_wids = selected_pid_wids[i]
        cur_label = selected_labels[i]

        for pid_wid in cur_pid_wids:
            selected_wells.append((pid_wid[0], pid_wid[1], int(cur_label)))

    # Group these wells by their pids
    selected_well_dict = {}
    for well in selected_wells:
        cur_pid, cur_wid, cur_label = well[0], well[1], well[2]

        if cur_pid in selected_well_dict:
            selected_well_dict[cur_pid].append((cur_wid, cur_label))
        else:
            selected_well_dict[cur_pid] = [(cur_wid, cur_label)]

    return selected_well_dict


def extract_instance(pid, wid, label, output_dir='./output'):
    """
    Extract all images in the given pid and wid pair (many fields of view).
    Then, it saves the extracted 5-channel images as a tensor in `output_dir`.

    Args:
        pid(int): plate id
        wid(int): well id
        label(int): 1 -> activated, 0 -> not activated
        output_dir(str): directory to store image tensors.
    """

    paths = [p.format(pid) for p in raw_paths]

    # Dynamically count number of sids for this pid-wid
    sid_files = [f for f in glob(paths[0]) if
                 re.search(r'^.*_{}_s\d_.*\.tif$'.format(wid), basename(f))]
    sid_num = len(sid_files)

    for sid in range(1, sid_num + 1):
        # Each sid generates one instance
        image_names, images = [], []

        for p in paths:
            # Search current pid-wid-sid
            cur_file = [f for f in glob(p) if
                        re.search(r'^.*_{}_s{}_.*\.tif$'.format(wid, sid),
                                  basename(f))]

            # We should only see one result returned from the filter
            if len(cur_file) != 1:
                error = "Found more than one file for {}-{}-{}.".format(
                    pid, wid, sid
                )
                raise ValueError(error)

            image_names.append(cur_file[0])

        # Read 5 channels
        for n in image_names:
            images.append(cv2.imread(n, -1) * 16)

        # Store each image as a 5 channel 3d matrix
        image_instance = np.array(images)

        # Save the instance with its label
        np.savez_compressed(join(output_dir, 'img_{}_{}_{}_{}.npz'.format(
            pid, wid, sid, label
        )), img=image_instance, )


def extract_plate(assay, pid, selected_well_dict):
    """
    Wrapper of `extract_instance()`. Uncompress zip files before
    extracting instances from one plate.

    Args:
        assay(int): assay index
        pid(int): plate id
        selected_well_dict(dict): {pid: [(wid, label), (wid, label)...]}
    """

    # output_dir should be unique for each pid, so they can be transferred
    # back from the cluster
    output_dir = './assay_{}_output_pid_{}'.format(assay, pid)

    if not exists(output_dir):
        mkdir(output_dir)

    for c in raw_channels:
        # Unzip the zip file and then remove it
        with zipfile.ZipFile("./{}-{}.zip".format(pid, c), 'r') as fp:
            fp.extractall('./')

        os.remove("./{}-{}.zip".format(pid, c))

    # Extract all instances from all selected wells in this plate
    for wid_tuple in selected_well_dict[pid]:
        extract_instance(pid, wid_tuple[0], wid_tuple[1], output_dir)

    # Clean up directories
    for c in raw_channels:
        rmtree("./{}-{}".format(pid, c))


if __name__ == '__main__':

    # Load meta data
    assay = int(argv[1])
    pid = int(argv[2])
    selected_well_dict = load(open('./selected_well_dict.json', 'r'))

    # Extract images from plates
    print("Starting to extract images from {} wells in plate {}".format(
        len(selected_well_dict[pid]),
        pid
    ))
    start_time = time.time()

    extract_plate(assay, pid, selected_well_dict)

    print("Finished extraction: {} seconds".format(time.time() - start_time))
